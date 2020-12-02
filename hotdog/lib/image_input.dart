import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:google_fonts/google_fonts.dart';

class ImageInput extends StatefulWidget {
  final Function onSelectImage;

  ImageInput(this.onSelectImage);

  @override
  _ImageInputState createState() => _ImageInputState();
}

class _ImageInputState extends State<ImageInput> {
  File _storedImage;
  final picker = ImagePicker();
  String resultText = '';
  bool isHotdog = false;
  bool isRecognized = false;

  Future<void> _takePicture() async {
    final imageFile = await picker.getImage(
      source: ImageSource.camera,
    );
    if (imageFile == null) {
      return;
    }
    setState(() {
      _storedImage = File(imageFile.path);
    });
    predictHotdog(File(imageFile.path));
  }

  Future<void> _getImageFromGallery() async {
    final imageFile = await picker.getImage(
      source: ImageSource.gallery,
    );
    if (imageFile == null) {
      return;
    }
    setState(() {
      _storedImage = File(imageFile.path);
    });
    predictHotdog(File(imageFile.path));
  }

  static Future loadModel() async {
    Tflite.close();
    try {
      await Tflite.loadModel(
          model: 'assets/hotdog.tflite', labels: 'assets/labels.txt');
    } on PlatformException {
      print("Failed to load the model");
    }
  }

  Future predictHotdog(File image) async {
    var recognition = await Tflite.runModelOnImage(
      path: image.path,
      imageMean: 117, // defaults to 117.0
      imageStd: 117, // defaults to 1.0
      numResults: 2, // defaults to 5
      threshold: 0.2, // defaults to 0.1
      asynch: true,
    );

    if (recognition.isNotEmpty) {
      if (recognition[0]["confidence"] < 0.5) {
        setState(() {
          isRecognized = true;
          isHotdog = true;
          resultText = "Hotdog";
        });
      } else {
        setState(() {
          isRecognized = true;
          isHotdog = false;
          resultText = "Not Hotdog";
        });
      }
    }
  }

  @override
  void initState() {
    super.initState();
    loadModel().then((val) {
      setState(() {});
    });
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
              child: _storedImage != null
                  ? Image.file(
                      _storedImage,
                      fit: BoxFit.cover,
                      width: double.infinity,
                    )
                  : Text(
                      'No Image Taken',
                      textAlign: TextAlign.center,
                    ),
            ),
            isRecognized
                ? Stack(
                    children: [
                      Container(
                        color: isHotdog ? Colors.green : Colors.red,
                        height: 80,
                        padding: EdgeInsets.all(10),
                        alignment: Alignment.topCenter,
                        child: Text(
                          "$resultText",
                          style: GoogleFonts.bungeeInline(
                            textStyle: TextStyle(
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
            Expanded(
              child: FlatButton.icon(
                icon: Icon(Icons.photo_camera),
                label: Text('カメラ'),
                textColor: Theme.of(context).primaryColor,
                onPressed: _takePicture,
              ),
            ),
            Expanded(
              child: FlatButton.icon(
                icon: Icon(Icons.photo_library),
                label: Text('ギャラリー'),
                textColor: Theme.of(context).primaryColor,
                onPressed: _getImageFromGallery,
              ),
            ),
          ],
        ),
      ],
    );
  }
}
