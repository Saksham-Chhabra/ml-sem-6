import * as tf from "@tensorflow/tfjs";

/**
 * McCulloch–Pitts Neuron
 * @param {number[]} inputs   Binary inputs (0 or 1)
 * @param {number[]} weights  Fixed weights
 * @param {number} threshold  Threshold value
 * @returns {number}          0 or 1
 */
function mpNeuron(inputs, weights, threshold) {
  return tf.tidy(() => {
    const x = tf.tensor1d(inputs);
    const w = tf.tensor1d(weights);

    // Weighted sum
    const netInput = tf.dot(x, w);

    // Step function
    const output = netInput.greaterEqual(threshold).cast("int32");

    return output.arraySync();
  });
}

mpNeuron([1,0,1],[1,1,1],2)