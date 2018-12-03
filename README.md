# Robust-CNN-with-Differential-Privacy

* Differential Privacy Noise Layer

```python
 def _dp_mult(self, sensitivity_norm, output_dim=None):
    dp_eps = self.hps.dp_epsilon
    dp_del = self.hps.dp_delta
    if sensitivity_norm == 'l2':
        # Use the Gaussian mechanism
        # attack_norm_bound = 0.1
        return self.hps.attack_norm_bound *  \
               math.sqrt(2 * math.log(1.25 / dp_del)) / dp_eps

    else:
        return 0
```

We only use Gaussian mechanism for the noise layer which is inserted after the first input layer of the CNN. This relys on the _sensitivity_ of the pre-noise layers(input layer in this case), function `g`. The sensitivity of the function `g` is defined as the maximum change in output that can be produced by a change in the input, given L-2 norm distance metrices for the input and output: 

![\Delta_{2, 2}^g = max_{x, x'} \frac{ \left \| g(x) - g(x') \right \|_2}{ \left \| x - x ' \right \|_2}](http://latex.codecogs.com/gif.latex?%5CDelta_%7B2%2C%202%7D%5Eg%20%3D%20max_%7Bx%2C%20x%27%7D%20%5Cfrac%7B%20%5Cleft%20%5C%7C%20g%28x%29%20-%20g%28x%27%29%20%5Cright%20%5C%7C_2%7D%7B%20%5Cleft%20%5C%7C%20x%20-%20x%20%27%20%5Cright%20%5C%7C_2%7D)

Assuming we can compute the sensitivity of the pre-noise layers, the noise layers leverages the Gaussian mechanisms as follows. On every invocation of the network on an input x (whether for training or prediction) the noise computes ![g(X) + Z](http://latex.codecogs.com/gif.latex?g%28X%29%20&plus;%20Z), where the coordinates Z = (Z1, Z2, ..., Zm) are independent random variables from a noise distribution defined by the function ![noise(\Delta , L, \epsilon, \delta)](http://latex.codecogs.com/gif.latex?noise%28%5CDelta%20%2C%20L%2C%20%5Cepsilon%2C%20%5Cdelta%29)


Above python function `_dp_mult` for pixeldp model is to calculate the noise, where it uses the Gaussian distribution with mean=0 and standard deviation ![\sigma = \sqrt{2 ln(\frac{1.25}{\delta})} \Delta_{2, 2} L / \epsilon](http://latex.codecogs.com/gif.latex?%5Csigma%20%3D%20%5Csqrt%7B2%20ln%28%5Cfrac%7B1.25%7D%7B%5Cdelta%7D%29%7D%20%5CDelta_%7B2%2C%202%7D%20L%20/%20%5Cepsilon); it gives ![(\epsilon, \delta)](http://latex.codecogs.com/gif.latex?%28%5Cepsilon%2C%20%5Cdelta%29-DP) for ![\epsilon \leq  1](http://latex.codecogs.com/gif.latex?%5Cepsilon%20%5Cleq%201)

Here, L denotes the 2-norm size of the attack against which the PixelDP network provides ![(\epsilon, \delta)](http://latex.codecogs.com/gif.latex?%28%5Cepsilon%2C%20%5Cdelta%29-DP) for ![\epsilon \leq  1](http://latex.codecogs.com/gif.latex?%5Cepsilon%20%5Cleq%201); we call it _construction attack bound__ (in the experiment, we set `attack_norm_bound = 0.1` )


  

