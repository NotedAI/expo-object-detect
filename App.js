import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  Dimensions,
  Image,
  Button,
  TouchableOpacity,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import "@tensorflow/tfjs-react-native";
import jpeg from "jpeg-js";

const { width } = Dimensions.get("window");

export default function App() {
  const [facing, setFacing] = useState("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [model, setModel] = useState(undefined);
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelJson = require("./assets/model/model.json");
        const modelWeights = require("./assets/model/group1-shard1of1.bin");

        await tf.ready();

        const model = await tf.loadGraphModel(
          bundleResourceIO(modelJson, modelWeights)
        );

        setModel(model);

        console.log("TF model loaded succefully");
      } catch (error) {
        console.error("Error loading the model:", error);
      }
    };
    loadModel();
  }, []);

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }
  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          We need your permission to show the camera
        </Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  const handleTakePic = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.3,
        base64: true,
      });
      // scale photo to 400*400
      const scaledPhoto = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 40, height: 40 } }],
        { base64: true }
      );

      convertImage2RGBArray(scaledPhoto.base64);
      console.log("done");
    }
  };
  const convertImage2RGBArray = (base64) => {
    const buffer = Buffer.from(base64, "base64");
    const rawImageData = jpeg.decode(buffer);
    const { data, width, height } = rawImageData;
    const rgbArray = [];
    for (let i = 0; i < height; i++) {
      const row = [];
      for (let j = 0; j < width; j++) {
        const idx = (i * width + j) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        // Ignore data[idx + 3] (alpha value)
        row.push([r, g, b]);
      }
      rgbArray.push(row);
    }

    const resultArray = [rgbArray];

    // const inputMean = 127.5;
    // const inputStd = 127.5;
    // resultArray = resultArray
    //   .sub(tf.scalar(inputMean))
    //   .div(tf.scalar(inputStd));

    // const res = model.predict(imageResized);
    // setPrediction(res);
    // console.log(prediction);

    console.log(resultArray[0]);
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />
      {/* <Image source={{ uri: capturedPhoto }} style={styles.capturedImage} /> */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => {
            setFacing((current) => (current === "back" ? "front" : "back"));
          }}
        >
          <Text style={styles.text}>Flip</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={handleTakePic}>
          <Text style={styles.text}>Take</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => {
            setCapturedPhoto(null);
            console.log("clear");
          }}
        >
          <Text style={styles.text}>Clear</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  camera: {
    width: width,
    height: width,
  },
  capturedImage: {
    width: 250,
    height: 250,
    marginTop: 5,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "transparent",
    marginTop: 5,
  },
  button: {
    flex: 1,
    alignItems: "center",
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "black",
  },
});
