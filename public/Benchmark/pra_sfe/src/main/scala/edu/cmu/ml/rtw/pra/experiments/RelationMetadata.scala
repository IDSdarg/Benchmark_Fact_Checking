package edu.cmu.ml.rtw.pra.experiments

import edu.cmu.ml.rtw.pra.data.Instance
import edu.cmu.ml.rtw.pra.data.NodeInstance
import edu.cmu.ml.rtw.pra.data.NodePairInstance
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.util.FileUtil
import edu.cmu.ml.rtw.pra.util.JsonHelper

import scala.collection.mutable

import org.json4s._

class RelationMetadata(
  params: JValue,
  praBase: String,
  outputter: Outputter,
  fileUtil: FileUtil = new FileUtil
) {
  implicit val formats = DefaultFormats

  val baseDir: String = params match {
    case JNothing => null
    case JString(path) if (path.startsWith("/")) => fileUtil.addDirectorySeparatorIfNecessary(path)
    case JString(name) => s"${praBase}relation_metadata/${name}/"
    case jval => {
      jval \ "name" match {
        case JString(name) => s"${praBase}relation_metadata/${name}/"
        case _ => {
          jval \ "directory" match {
            case JString(dir) => dir
            case _ => {
              outputter.warn("Couldn't find a base directory for relation metadata...")
              null
            }
          }
        }
      }
    }
  }

  // var unallowedPairs: mutable.HashSet[(Int, Int, Int)] = new mutable.HashSet[(Int, Int, Int)]


  def getUnallowedEdges(relation: String, graph: Graph): Seq[Int] = {
    val unallowedEdges = new mutable.ArrayBuffer[Int]
    val rel_type = 671
    if (!graph.hasEdge(relation)) return unallowedEdges.toSeq

    // The relation itself is an unallowed edge type.
    val relIndex = graph.getEdgeIndex(relation)
    unallowedEdges += relIndex
    unallowedEdges += rel_type
    unallowedEdges.toSeq
  }

}

object RelationMetadata {
  val empty = new RelationMetadata(JNothing, "/dev/null", Outputter.justLogger)
}
