import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs/promises";

const imageToTest = "assets/image3.jpg";

Promise.all([cocoSsd.load(), fs.readFile(imageToTest)]).then((results) => {
    const model = results[0];
    const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3);
    return model.detect(imgTensor as tf.Tensor3D);
}).then((predictions) => {
    console.log(JSON.stringify(predictions, null, 2));
});
