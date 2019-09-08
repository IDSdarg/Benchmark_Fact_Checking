package edu.cmu.ml.rtw.pra.experiments

import edu.cmu.ml.rtw.pra.pipeline.Step
import edu.cmu.ml.rtw.pra.util.FileUtil
import edu.cmu.ml.rtw.pra.util.JsonHelper
import edu.cmu.ml.rtw.pra.util.Pair
import edu.cmu.ml.rtw.pra.util.SpecFileReader

import edu.cmu.ml.rtw.pra.data.Split

import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.graphs.GraphCreator
import edu.cmu.ml.rtw.pra.graphs.GraphOnDisk

import edu.cmu.ml.rtw.pra.operations.Operation

import scala.collection.mutable

import org.json4s._
import org.json4s.native.JsonMethods.{pretty,render,parse}

// This class has two jobs.  This first is to create all of the necessary input files from the
// parameter specification (e.g., create actual graph files from the relation sets that are
// specified in the parameters).  This part just looks at the parameters and creates things on the
// filesystem.
//
// The second job is to create all of the (in-memory) objects necessary for running the code, then
// run it.  The general design paradigm for this is that there should be one object per parameter
// key (e.g., "operation", "relation metadata", "split", "graph", etc.).  At each level, the object
// creates all of the sub-objects corresponding to the parameters it has, then performs its
// computation.  This Driver is the top-level object, and its main computation is an Operation.
//
// TODO(matt): Refactor this to make use of the pipeline architecture for the graph, embedding,
// split, and other required input files.
class Driver(
  praBase: String,
  methodName: String,
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Driver"

  val outputter = new Outputter(params \ "output", praBase, methodName, fileUtil)
  override val inProgressFile = outputter.baseDir + "in_progress"
  override val paramFile = outputter.baseDir + "params.json"

  // TODO(matt): this will eventually include the split, embeddings, and whatever else.
  override val inputs = getGraphInput(params \ "graph")

  // TODO(matt): define this correctly.
  override val outputs = Set[String]()

  override def _runStep() {
    val baseKeys = Seq("graph", "split", "relation metadata", "operation", "output")

    JsonHelper.ensureNoExtras(params, "base", baseKeys)
    
    // We create the these auxiliary input files first here, because we allow a "no op" operation,
    // which means just create all of the generated input files and then quit.  But we have to do
    // this _after_ we create the output directory with outputter.begin(), so that two experiments
    // needing the same graph won't both try to create it, or think that it's done while it's still
    // being made.  We'll delete the output directory in the case of a no op.
    outputter.begin()
    

    val relationMetadata =
      new RelationMetadata(params \ "relation metadata", praBase, outputter, fileUtil)
    
    val split = Split.create(params \ "split", praBase, outputter, fileUtil)

    val graph = Graph.create(params \ "graph", praBase + "/graphs/", outputter, fileUtil)

    val operation =
      Operation.create(params \ "operation", graph, split, relationMetadata, outputter, fileUtil)


    val start_time = System.currentTimeMillis
   
    outputter.info("\n\n\n\nRunning PRA for relation " + (params \ "split" \ "name").extract[String])
    outputter.setRelation((params \ "split" \ "test_file").extract[String], (params \ "split" \ "name").extract[String])

    operation.runRelation((params \ "split" \ "name").extract[String])
   
    val end_time = System.currentTimeMillis
    val millis = end_time - start_time
    var seconds = (millis / 1000).toInt
    val minutes = seconds / 60
    seconds = seconds - minutes * 60
    outputter.logToFile("PRA appears to have finished all relations successfully\n")
    outputter.logToFile(s"Total time: $minutes minutes and $seconds seconds\n")
    outputter.info(s"Total time: $minutes minutes and $seconds seconds")
  }

  def getGraphInput(graphParams: JValue): Set[(String, Option[Step])] = {
    var graphName = ""
    var paramsSpecified = false
    // First, is this just a path, or do the params specify a graph name?  If it's a path, we'll
    // just use the path as is.  Otherwise, we have some processing to do.
    graphParams match {
      case JNothing => {}
      case JString(path) if (path.startsWith("/")) => {
        if (!fileUtil.fileExists(path)) {
          throw new IllegalStateException("Specified path to graph does not exist!")
        }
      }
      case JString(name) => graphName = name
      case jval => {
        jval \ "name" match {
          case JString(name) => {
            graphName = name
            paramsSpecified = true
          }
          case other => { }
        }
      }
    }
    if (graphName != "") {
      val graphDir = s"${praBase}graphs/"

      if (paramsSpecified) {
        val creator = new GraphCreator(s"${praBase}", graphParams, outputter, fileUtil)

        Set((graphDir, Some(creator)))

      } else {
        Set((graphDir, None))
      }
    } else {
      Set()
    }
  }



}
