import React, { useEffect, useState } from "react";
import {
  View,
  Image,
  StyleSheet,
  Dimensions,
  Text,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import { fetch, decodeJpeg } from "@tensorflow/tfjs-react-native";
import Svg, { Rect, Text as SvgText } from "react-native-svg";

const App = () => {
  const [imageUri, setImageUri] = useState(
    "http://192.168.0.4:8000/test02.jpg"
  );
  const [boxesData, setBoxesData] = useState([]);
  const [classesData, setClassesData] = useState([]);
  const [scoresData, setScoresData] = useState([]);
  const [loadingTime, setLoadingTime] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const modelJson = require("./assets/graph_model/model.json");
      const modelWeights = require("./assets/graph_model/combined_output.bin");
      const model = await tf.loadGraphModel(
        bundleResourceIO(modelJson, modelWeights)
      );
      console.log("model loaded");

      const response = await fetch(imageUri, {}, { isBinary: true });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const startTime = performance.now();
      const imageData = await response.arrayBuffer();

      let imageTensor = decodeJpeg(new Uint8Array(imageData));
      imageTensor = tf.image.resizeBilinear(imageTensor, [320, 320]);
      imageTensor = imageTensor.slice([0, 0, 0], [-1, -1, 3]);
      imageTensor = imageTensor.expandDims(0);
      console.log("image trimed");

      const predictions = await model.executeAsync(imageTensor);

      console.log("prediction done");

      const boxesData = predictions[7].arraySync()[0].slice(0, 10);
      const scoresData = predictions[5].arraySync()[0].slice(0, 10);
      const classesData = predictions[2].arraySync()[0].slice(0, 10);

      setBoxesData(boxesData);
      setScoresData(scoresData);
      setClassesData(classesData);

      const endTime = performance.now();
      const totalTime = (endTime - startTime).toFixed(2);
      setLoadingTime(totalTime);
    };

    loadModel();
  }, []);

  const wholeWindowWidth = Dimensions.get("window").width;
  const wholeWindowHeight = Dimensions.get("window").height;
  const windowWidth = 320;
  const windowHeight = 320;

  return (
    <View style={styles.container}>
      <Image source={{ uri: imageUri }} style={styles.image} />
      <Svg
        height={wholeWindowHeight}
        width={wholeWindowWidth}
        style={StyleSheet.absoluteFill}
      >
        {boxesData.map((boxes, index) => {
          const [ymin, xmin, ymax, xmax] = boxes;
          const width = (xmax - xmin) * windowWidth;
          const height = (ymax - ymin) * windowHeight;
          const left = xmin * windowWidth;
          const top = ymin * windowHeight;

          if (scoresData[index] > 0.5 && scoresData[index] <= 1) {
            return (
              <React.Fragment key={index}>
                <Rect
                  x={left}
                  y={top}
                  width={width}
                  height={height}
                  stroke="red"
                  strokeWidth="2"
                  fill="none"
                />
                <SvgText
                  x={left}
                  y={top - 5}
                  fill="red"
                  fontSize="8"
                  fontWeight="bold"
                >
                  {classesData[index]}: {scoresData[index]}
                </SvgText>
              </React.Fragment>
            );
          }
        })}
      </Svg>
      <Text style={styles.time}>Total loading time: {loadingTime} ms</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 100,
  },
  image: {
    ...StyleSheet.absoluteFillObject,
    width: 320,
    height: 320,
  },
  time: {
    color: "blue",
    fontSize: 14,
    margin: 10,
  },
});

export default App;
