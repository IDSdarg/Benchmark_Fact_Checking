package edu.cmu.ml.rtw.pra.operations

import edu.cmu.ml.rtw.pra.data.Dataset
import edu.cmu.ml.rtw.pra.data.Instance
import edu.cmu.ml.rtw.pra.data.Split
import edu.cmu.ml.rtw.pra.data.NodePairInstance
import edu.cmu.ml.rtw.pra.data.NodePairSplit
import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.experiments.RelationMetadata
import edu.cmu.ml.rtw.pra.features.FeatureGenerator
import edu.cmu.ml.rtw.pra.features.FeatureMatrix
import edu.cmu.ml.rtw.pra.features.MatrixRow
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.models.BatchModel
import edu.cmu.ml.rtw.pra.models.LogisticRegressionModel


import edu.cmu.ml.rtw.pra.util.Dictionary
import edu.cmu.ml.rtw.pra.util.FileUtil
import edu.cmu.ml.rtw.pra.util.JsonHelper
import edu.cmu.ml.rtw.pra.util.MutableConcurrentDictionary

import scala.collection.JavaConverters._
import scala.collection.concurrent
import scala.collection.mutable

import org.json4s._
import org.json4s.JsonDSL._

trait Operation[T <: Instance] {
  def runRelation(relation: String)
}

object Operation {
  // I had this returning an Option[Operation[T]], and I liked that, because it removes the need
  // for NoOp.  However, it gave me a compiler warning about inferred existential types, and
  // leaving it without the Option doesn't give me that warning.  So, I'll take the slightly more
  // ugly design instead of the compiler warning.
  def create[T <: Instance](
    params: JValue,
    graph: Option[Graph],
    split: Split[T],
    relationMetadata: RelationMetadata,
    outputter: Outputter,
    fileUtil: FileUtil
  ): Operation[T] = new TrainAndTest(params, graph, split, relationMetadata, outputter, fileUtil)
  
}


class TrainAndTest[T <: Instance](
  params: JValue,
  graph: Option[Graph],
  split: Split[T],
  relationMetadata: RelationMetadata,
  outputter: Outputter,
  fileUtil: FileUtil
) extends Operation[T] {
  val paramKeys = Seq("type", "features", "learning")
  JsonHelper.ensureNoExtras(params, "operation", paramKeys)

  override def runRelation(relation: String) {

    // First we get features.
    val generator = FeatureGenerator.create(
      params \ "features",
      graph,
      split,
      relation,
      relationMetadata,
      outputter,
      fileUtil = fileUtil
    )

    val trainingData = split.getTrainingData(graph)
    val trainingMatrix = generator.createTrainingMatrix(trainingData)
    outputter.outputFeatureMatrix(true, trainingMatrix, generator.getFeatureNames())

    // Then we train a model.
    val model = BatchModel.create(params \ "learning", split, outputter)
    val featureNames = generator.getFeatureNames()
    model.train(trainingMatrix, trainingData, featureNames)

    // Then we test the model.
    val testingData = split.getTestingData(graph)
    val testMatrix = generator.createTestMatrix(testingData)
    outputter.outputFeatureMatrix(false, testMatrix, generator.getFeatureNames())
    val scores = model.classifyInstances(testMatrix)
    outputter.outputScores(scores, trainingData)

  }
}




