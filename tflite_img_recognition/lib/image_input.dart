import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:google_fonts/google_fonts.dart';
import './classifier.dart';

class ImageInput extends StatefulWidget {
  final Function onSelectImage;

  const ImageInput(this.onSelectImage);

  @override
  _ImageInputState createState() => _ImageInputState();
}

class _ImageInputState extends State<ImageInput> {
  // 取得した画像ファイル
  File? _storedImage;
  // ImagePickerのインスタンス
  final picker = ImagePicker();
  // 推論結果のテキスト
  String resultText = '';
  // ホットドックかどうか
  bool isHotdog = false;
  // 推論済みかどうか
  bool isPredicted = false;
  // Classifierのインストラクタ
  late Classifier _classifier;

  /* カメラから画像を取得 */
  Future<void> _takePicture() async {
    final imageFile = await picker.pickImage(
      source: ImageSource.camera,
    );
    if (imageFile == null) {
      return;
    }
    setState(() {
      _storedImage = File(imageFile.path);
    });
    predict();
  }

  /* ギャラリーから画像を取得 */
  Future<void> _getImageFromGallery() async {
    final imageFile = await picker.pickImage(
      source: ImageSource.gallery,
    );
    if (imageFile == null) {
      return;
    }
    setState(() {
      _storedImage = File(imageFile.path);
    });
    predict();
  }

  /* initState */
  @override
  void initState() {
    super.initState();
    _classifier = Classifier('hotdog.tflite');
  }

  /* 推論処理 */
  void predict() async {
    // classifierへの入力はImage型なので、Image型にデコード
    img.Image inputImage = img.decodeImage(_storedImage!.readAsBytesSync())!;
    // 推論を行う
    double confidence = _classifier.predict(inputImage);

    // 推論結果から識別を行う
    if (confidence < 0.5) {
      setState(() {
        isPredicted = true;
        isHotdog = true;
        resultText = "Hotdog";
      });
    } else {
      setState(() {
        isPredicted = true;
        isHotdog = false;
        resultText = "Not Hotdog";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final Size size = MediaQuery.of(context).size;
    return Column(
      children: [
        Stack(
          children: [
            Container(
              width: size.width,
              height: 480,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                border: Border.all(width: 1, color: Colors.grey),
              ),
              child: _storedImage == null
                  ? const Text(
                      "No Image Taken",
                      textAlign: TextAlign.center,
                    )
                  : Image.file(
                      _storedImage!,
                      fit: BoxFit.cover,
                      width: double.infinity,
                    ),
            ),
            isPredicted
                ? Stack(
                    children: [
                      Container(
                        color: isHotdog ? Colors.green : Colors.red,
                        height: 80,
                        padding: const EdgeInsets.all(10),
                        alignment: Alignment.topCenter,
                        child: Text(
                          resultText,
                          style: GoogleFonts.bungeeInline(
                            textStyle: const TextStyle(
                              fontSize: 40,
                              fontWeight: FontWeight.normal,
                            ),
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ),
                      Container(
                        height: 120,
                        alignment: Alignment.bottomCenter,
                        child: CircleAvatar(
                          maxRadius: 35,
                          backgroundColor: isHotdog ? Colors.green : Colors.red,
                          child: Icon(
                            isHotdog ? Icons.check : Icons.clear,
                            size: 50,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  )
                : Container(),
          ],
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            OutlinedButton.icon(
              icon: const Icon(Icons.photo_camera),
              label: const Text('カメラ'),
              onPressed: _takePicture,
            ),
            OutlinedButton.icon(
              icon: const Icon(Icons.photo_library),
              label: const Text('ギャラリー'),
              onPressed: _getImageFromGallery,
            ),
          ],
        ),
      ],
    );
  }
}
