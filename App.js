import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  Dimensions,
  Image,
  TouchableOpacity,
} from "react-native";
import { CameraView } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import "@tensorflow/tfjs-react-native";

const { width } = Dimensions.get("window");

export default function App() {
  const [facing, setFacing] = useState("back");
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

  const toggleCameraFacing = () => {
    setFacing((current) => (current === "back" ? "front" : "back"));
  };

  const toggleTakePic = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      setCapturedPhoto(photo.uri);
      processImage(photo.uri);
    }
  };

  const processImage = async (uri) => {
    const response = await fetch(uri, {}, { isBinary: true });
    const imageData = await response.arrayBuffer();
    const uint8Array = new Uint8Array(imageData);

    // Decode the image data into a tensor
    const imageTensor = decodeJpeg(uint8Array);
    
    const [height, width] = model.inputs[0].shape.slice(1, 3);
    let imageResized = tf.image.resizeBilinear(imageTensor, [height, width]);
    imageResized = imageResized.expandDims(0);
    
    if (model.inputs[0].dtype === 'float32') {
      const inputMean = 127.5;
      const inputStd = 127.5;
      imageResized = imageResized.sub(tf.scalar(inputMean)).div(tf.scalar(inputStd));
    }

    console.log("ready to predict");
    const res = model.predict(imageResized);
    setPrediction(res);
    console.log(prediction);
  };

  const toggleClear = () => {
    setCapturedPhoto(null);
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />
      {/* <Image source={{ uri: capturedPhoto }} style={styles.capturedImage} /> */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
          <Text style={styles.text}>Flip</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={toggleTakePic}>
          <Text style={styles.text}>Take</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={toggleClear}>
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
    width: width,
    height: width,
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
