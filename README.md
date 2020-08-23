# nannernest

![Python package](https://github.com/EthanRosenthal/nannernest/workflows/Python%20package/badge.svg?branch=master)

A small package for optimizing banana coverage on peanut butter and banana sandwiches.


![assets/perfect_sandwich.jpg](assets/perfect_sandwich.jpg)


## Installation

`nannernest` is generally pip installable. Due to some C dependencies with the nesting library that I use [nest2D](https://github.com/markfink/nest2D), along with an outstanding [PR](https://github.com/markfink/nest2D/pull/2), I would recommend the following way to install everything:

 First, make sure you have [boost](https://www.boost.org/) and [cmake](https://cmake.org/) installed. If you are on Linux, then you may have `cmake` installed, and you can install `boost` with 
 
 ```commandline
sudo apt-get install libboost-all-dev 
```
 
 I'm on a Mac, and I installed `cmake` with conda and `boost` with brew:
 
 ```commandline
conda install cmake
brew install boost
```

Next, pip install my fork of `nest2D`:

```commandline
pip install git+https://github.com/EthanRosenthal/nest2D.git@download-dependencies
```

Finally, pip install `nannernest`

```commandline
pip install nannernest
```

## Usage

Take a top-down picture that contains your banana and at least one slice of bread. Pass the image in via command line:

```commandline
$ nannernest my_image.jpg
```

### CLI Details

```commandline
$ nannernest --help
Usage: nannernest [OPTIONS] IMAGE_PATH

Arguments:
  IMAGE_PATH  Image file which contains bread and banana  [required]

Options:
  --num-slices INTEGER            Maxmimum number of banana slices to consider
                                  [default: 16]

  --banana-pct FLOAT              Percent of banana to cut up  [default: 75]
  --mask-threshold FLOAT          Threshold of segmentation mask.  [default:
                                  0.6]

  --peel-scaler FLOAT             Fraction of slice that is assumed to belong
                                  to banana insides versus the peel.
                                  [default: 0.8]

  --ellipse-ratio FLOAT           Assumed ratio of minor axis to major axis of
                                  banana slice ellipses  [default: 0.85]

  --plot-segmentation / --no-plot-segmentation
                                  Whether or not to plot the segmentation
                                  masks  [default: False]

  --plot-slicing / --no-plot-slicing
                                  Whether or not to plot the slicing circle
                                  and skeleton  [default: False]

  --output TEXT                   Name of file to output  [default:
                                  perfect_sandwich.jpg]
```
