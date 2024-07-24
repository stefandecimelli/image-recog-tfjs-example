import cocoSsd from "@tensorflow-models/coco-ssd"
import tf from "@tensorflow/tfjs-node"
import fs from "fs/promises"

Promise.all([cocoSsd.load(), fs.readFile("assets/image1.jpg")]).then((results => {
	const model = results[0];
	const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3);
	return model.detect(imgTensor);
})).then((predictions) => {
	console.log(JSON.stringify(predictions, null, 2))
});
