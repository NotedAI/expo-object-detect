import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  Dimensions,
  Button,
  TouchableOpacity,
} from "react-native";
import { CameraView } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import "@tensorflow/tfjs-react-native";

const { width } = Dimensions.get("window");

export default function App() {
  const [facing, setFacing] = useState("front");
  const [model, setModel] = useState(undefined);
  const cameraRef = useRef(null);

  // const [permission, requestPermission] = useCameraPermissions();

  // if (!permission) {
  //   return <View />;
  // }
  // if (!permission.granted) {
  //   // Camera permissions are not granted yet.
  //   return (
  //     <View style={styles.container}>
  //       <Text style={{ textAlign: "center" }}>
  //         We need your permission to show the camera
  //       </Text>
  //       <Button onPress={requestPermission} title="grant permission" />
  //     </View>
  //   );
  // }

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
      const imageTensor = decodeJpeg(
        tf.util.encodeString(photo.base64, "base64"),
        3
      );

      // Normalize the image tensor values to float
      const normalizedTensor = imageTensor.div(tf.scalar(255));

      console.log("Image captured and normalized:", normalizedTensor);
      const prediction = model.predict(normalizedTensor);
      console.log("Prediction: ", prediction);
    }
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={toggleTakePic}>
          <Text style={styles.text}>Take</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
          <Text style={styles.text}>Flip</Text>
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
  buttonContainer: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "transparent",
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: "flex-end",
    alignItems: "center",
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "black",
  },
});
