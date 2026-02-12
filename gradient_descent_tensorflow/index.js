require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-node");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100,
});

regression.train();
regression.test(testFeatures, testLabels);

console.log("features", regression.features.dataSync());

plot({
  x: regression.bHistory,
  y: regression.mseHistory.reverse(),
  xLabel: "Value of b",
  yLabel: "Mean Squared Error",
});
