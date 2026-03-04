import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);
function makeContext(product, users) {
    const ages = users.map(u => u.age);
    const price = product.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const minPrice = Math.min(...price);
    const maxPrice = Math.max(...price);

    const colors =[...new Set(product.map(p => p.color))];
    const categories = [...new Set(product.map(p => p.category))];
    
    const colorIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        })) 
    
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {}
    const ageCounts = {}
    
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    })
    const productAvgAgeNorm = Object.fromEntries(
        product.map(product => {
            const avg = ageCounts[product.name] ?
            ageSums[product.name] / ageCounts[product.name] :
            midAge
            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )
    
    
    return {product,
        users,
        colorIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age colors + categories
        dimentions: 2 + colors.length + categories.length,
    }
}
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct(product, context) {
    // normalizando dados para ficar de 0 a 1
    // aplicar peso na recomendacao
    const price = tf.tensor1d([
        normalize(product.price, 
            context.minPrice, 
            context.maxPrice
        ) * WEIGHTS.price
    ]);
    
    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ]);
    
    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );
    
    const color = oneHotWeighted(
        context.colorIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );
    const test = tf.concat(
        [price, age, category, color]
    )
    console.log('Encoded product tensor:', test.dataSync());
    return tf.concat(
        [price, age, category, color]
    );
}

function encodeUser(user, context) {
    // New tensor
    if(user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context))
            )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }
};
        
function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users.forEach(user => {
        const userVector = encodeUser(user, context).dataSync()
        context.product.forEach(product => {
            const productVector = encodeProduct(product, context)
            .dataSync()
            const label = user.purchases.some(
                p => p.name === product.name ?
                1 : 
                0
            )
        inputs.push([...userVector, ...productVector])
        labels.push(label)
        })
    })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2,
        // size = userVector + productVector
    }
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const product = await((await fetch('/data/products.json')).json());
    // const users = await((await fetch('/data/users.json')).json());
    
    // Map 
    const context = makeContext(product, users);

    // transform to tensors in TensorFlow.js
    context.productVectors = product.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    });
    const trainData = createTrainingData(context);
    debugger;
    _globalCtx = context;

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
