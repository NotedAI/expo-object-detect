import React, { useState, useEffect } from "react";
import { StatusBar } from "expo-status-bar";
import { Platform, StyleSheet, Text, View, Dimensions } from "react-native";
import { Camera } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import "@tensorflow/tfjs-react-native";

export default function App() {
  const [textContent, setTextContent] = useState("Loading...");

  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelJson = require("./assets/model/model.json");
        const modelWeights = require("./assets/model/group1-shard1of1.bin");

        await tf.ready();

        const model = await tf.loadGraphModel(
          bundleResourceIO(modelJson, modelWeights)
        );

        console.log("TF model loaded succefully");

        
        setTextContent("TF Lite model loaded successfully.");
      } catch (error) {
        console.error("Error loading the model:", error);
        setTextContent("Failed to load the TF Lite model.");
      }
    };

    loadModel();
  }, []);

  return (
    <View style={styles.container}>
      <Text>Hello world</Text>
      <Text>{textContent}</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});
