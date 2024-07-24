import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import labels from "./labels";

const modelUrl = 'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';
const imagePath = "./assets/image3.jpg"
let model: tf.GraphModel;

const loadModel = async function () {
    console.log(`loading model from ${modelUrl}`);
    model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
    return model;
}

const processInput = (imagePath: string) => {
    console.log(`preprocessing image ${imagePath}`);

    const image = fs.readFileSync(imagePath);
    const buf = Buffer.from(image);
    const uint8array = new Uint8Array(buf);

    return tf.node.decodeImage(uint8array, 3).expandDims();
}

const runModel = (inputTensor: tf.Tensor) => {
    console.log('runnning model');

    return model.executeAsync(inputTensor);
}

const extractClassesAndMaxScores = (predictionScores: tf.Tensor<tf.Rank>) => {
    console.log('calculating classes & max scores');

    const scores = predictionScores.dataSync();
    const numBoxesFound = predictionScores.shape[1]!;
    const numClassesFound = predictionScores.shape[2]!;

    const maxScores = [];
    const classes = [];

    for (let i = 0; i < numBoxesFound; i++) {
        let maxScore = -1;
        let classIndex = -1;

        for (let j = 0; j < numClassesFound; j++) {
            if (scores[i * numClassesFound + j] > maxScore) {
                maxScore = scores[i * numClassesFound + j];
                classIndex = j;
            }
        }

        maxScores[i] = maxScore;
        classes[i] = classIndex;
    }

    return [maxScores, classes];
}

const maxNumBoxes = 5;

const calculateNMS = (outputBoxes: tf.Tensor<tf.Rank>, maxScores: number[]) => {
    console.log('calculating box indexes');

    const boxes = tf.tensor2d(outputBoxes.dataSync(), [outputBoxes.shape[1] as number, outputBoxes.shape[3] as number]);
    const indexTensor = tf.image.nonMaxSuppression(boxes, maxScores, maxNumBoxes, 0.5, 0.5);

    return indexTensor.dataSync();
}

let height = 1;
let width = 1;

const createJSONresponse = function (boxes: Float32Array | Int32Array | Uint8Array, scores: number[], indexes: Float32Array | Int32Array | Uint8Array, classes: number[]) {
    console.log('create JSON output');

    const count = indexes.length;
    const objects = [];

    for (let i = 0; i < count; i++) {
        const bbox = [];

        for (let j = 0; j < 4; j++) {
            bbox[j] = boxes[indexes[i] * 4 + j];
        }

        const minY = bbox[0] * height;
        const minX = bbox[1] * width;
        const maxY = bbox[2] * height;
        const maxX = bbox[3] * width;

        objects.push({
            bbox: [minX, minY, maxX, maxY],
            label: labels[classes[indexes[i]]],
            score: scores[indexes[i]]
        });
    }

    return objects;
}

const processOutput = (prediction: tf.Tensor<tf.Rank>[]) => {
    console.log('processOutput');

    const [maxScores, classes] = extractClassesAndMaxScores(prediction[0]);
    const indexes = calculateNMS(prediction[1], maxScores);

    return createJSONresponse(prediction[1].dataSync(), maxScores, indexes, classes);
}

loadModel().then(() => {
    const inputTensor = processInput(imagePath);
    height = inputTensor.shape[1] as number;
    width = inputTensor.shape[2] as number;
    return runModel(inputTensor);
}).then((prediction) => {
    const output = processOutput(prediction as tf.Tensor<tf.Rank>[]);
    console.log(output);
})
