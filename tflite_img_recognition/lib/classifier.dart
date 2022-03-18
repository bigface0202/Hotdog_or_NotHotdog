import 'dart:math';
import 'package:image/image.dart';
import 'package:collection/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class Classifier {
  // 推論エンジン
  late Interpreter _interpreter;
  // 推論用のオプション
  // 例えば推論に使うCPUのスレッド数やAndroid/iOS用の特殊なライブラリの使用などを指定できる
  late InterpreterOptions _interpreterOptions;

  // 入力画像サイズ
  late List<int> _inputShape;
  // 出力画像サイズ
  late List<int> _outputShape;

  // 出力結果格納バッファ
  late TensorBuffer _outputBuffer;

  // 入力の型
  late TfLiteType _inputType;
  // 出力の型
  late TfLiteType _outputType;

  // 重みファイル名
  late String _modelName;

  // 予測ラベルファイル名
  late String _labelFileName;
  // 予測ラベル格納用のList
  late List<String> _labels;

  late SequentialProcessor _probabilityProcessor;

  // 前処理に使用する正規化オプション
  final NormalizeOp _preProcessNormalizeOp = NormalizeOp(0, 1);
  // 推論後の後処理に使用する正規化オプション
  final NormalizeOp _postProcessNormalizeOp = NormalizeOp(0, 1);

  /* コンストラクタ */
  Classifier(String modelName, String labelName) {
    _modelName = modelName;
    _labelFileName = labelName;
    _interpreterOptions = InterpreterOptions();
    _interpreterOptions.threads = 1;

    loadModel();
    loadLabels();
  }

  /* モデルのロード */
  Future<void> loadModel() async {
    try {
      _interpreter =
          await Interpreter.fromAsset(_modelName, options: _interpreterOptions);

      _inputShape = _interpreter.getInputTensor(0).shape;
      _inputType = _interpreter.getInputTensor(0).type;
      _outputShape = _interpreter.getOutputTensor(0).shape;
      _outputType = _interpreter.getOutputTensor(0).type;

      _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
      // _probabilityProcessor =
      //     TensorProcessorBuilder().add(_postProcessNormalizeOp).build();
      print('Successfully model file is loaded.');
    } catch (e) {
      print('Something is happened during loading the models: $e');
    }
  }

  /* ラベルのロード */
  Future<void> loadLabels() async {
    try {
      _labels = await FileUtil.loadLabels(_labelFileName);
      print('Successfully label file is loaded.');
    } catch (e) {
      print('Something is happened during loading the labels: $e');
    }
  }

  /* 画像の前処理 */
  TensorImage preProcess(TensorImage inputImage) {
    // クロップサイズの指定
    // 入力画像の高さと幅のうち、小さい方が入力画像のクロップサイズとなる
    int cropSize = min(inputImage.height, inputImage.width);

    // 画像の前処理を行う
    TensorImage processedImage = ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(_preProcessNormalizeOp)
        .build()
        .process(inputImage);

    return processedImage;
  }

  /* 推論処理 */
  double predict(Image image) {
    // 入力の型を使ってTensorImageを初期化
    TensorImage inputImage = TensorImage(_inputType);

    // 画像をロード
    inputImage.loadImage(image);
    // inputImageに対して前処理を行う
    inputImage = preProcess(inputImage);

    // 推論時間を測る（別になくても良い）
    final predictStart = DateTime.now().millisecondsSinceEpoch;
    // 推論処理
    _interpreter.run(inputImage.buffer, _outputBuffer.getBuffer());
    // 推論時間の算出
    final predictTime = DateTime.now().millisecondsSinceEpoch - predictStart;

    print('Time to predict image: $predictTime ms');
    return _outputBuffer.getDoubleList()[0];
  }

  /* 推論エンジンのDestroy */
  void close() {
    _interpreter.close();
  }
}

MapEntry<String, double> getTopProbability(Map<String, double> labeledProb) {
  // @todo
  // PriorityQueueとはなんぞや
  // MapEntryとはなんぞや
  var pq = PriorityQueue<MapEntry<String, double>>(compare);
  pq.addAll(labeledProb.entries);

  return pq.first;
}

int compare(MapEntry<String, double> e1, MapEntry<String, double> e2) {
  if (e1.value > e2.value) {
    return -1;
  } else if (e1.value == e2.value) {
    return 0;
  } else {
    return 1;
  }
}
