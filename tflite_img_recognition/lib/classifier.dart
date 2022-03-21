import 'dart:math';
import 'package:image/image.dart';
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
  late final String _modelName;

  // 前処理に使用する正規化オプション
  final NormalizeOp _preProcessNormalizeOp = NormalizeOp(0, 1);

  /* コンストラクタ */
  Classifier(this._modelName) {
    _interpreterOptions = InterpreterOptions();
    _interpreterOptions.threads = 1;

    loadModel();
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
      print('Successfully model file is loaded.');
    } catch (e) {
      print('Something is happened during loading the models: $e');
    }
  }

  /* 画像の前処理 */
  TensorImage preProcess(TensorImage inputImage) {
    // クロップサイズの指定
    // 入力画像の高さと幅のうち、小さい方が入力画像のクロップサイズとなる
    int cropSize = min(inputImage.height, inputImage.width);

    // 画像の前処理を行う
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(_preProcessNormalizeOp)
        .build()
        .process(inputImage);
  }

  /* 推論処理 */
  double predict(Image image) {
    // 入力の型を使ってTensorImageを初期化
    TensorImage inputImage = TensorImage(_inputType);

    // 画像をロード
    inputImage.loadImage(image);
    // inputImageに対して前処理を行う
    inputImage = preProcess(inputImage);

    // 推論処理
    _interpreter.run(inputImage.buffer, _outputBuffer.getBuffer());
    return _outputBuffer.getDoubleList()[0];
  }

  /* 推論エンジンのDestroy */
  void close() {
    _interpreter.close();
  }
}
