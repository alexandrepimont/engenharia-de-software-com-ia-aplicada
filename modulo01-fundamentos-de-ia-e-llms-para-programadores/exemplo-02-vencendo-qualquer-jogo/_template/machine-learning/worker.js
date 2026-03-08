importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENTIONS = 640

let _labels = []
let _model = null
async function loadModelAndLabels() {
    await tf.ready()

    _labels = await (await fetch(LABELS_PATH)).json()
    _model = await tf.loadGraphModel(MODEL_PATH)

    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)
    tf.dispose(dummyInput)
    
    postMessage({type: 'model-loaded'})
}

// not possible to do async, tf function
function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input)
        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
            .div(255)
            .expandDims(0)

    })
}

async function runInference(tensor) {
    const output = await _model.executeAsync(tensor)
    tf.dispose(tensor) // free memory

    // caixas, pontuacoes, e classes
    const [boxes, scores, classes] = output.slice(0, 3)
    const [boxesData, scoresData, classesData] = await Promise.all(
        [
            boxes.data(),
            scores.data(),
            classes.data(),
        ]
    )
    output.forEach(t => tf.dispose(t)) // free memory
    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData
    }
}
loadModelAndLabels()


self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model) return

    const input = preprocessImage(data.image)
    const {width, height} = data.image
    
    const inferenceResults = await runInference(input)
    debugger
    
    postMessage({
        type: 'prediction',
        x: 400,
        y: 400,
        score: 0
    });


};

console.log('🧠 YOLOv5n Web Worker initialized');
