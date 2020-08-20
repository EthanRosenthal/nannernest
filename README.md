# nannernest

![Python package](https://github.com/EthanRosenthal/nannernest/workflows/Python%20package/badge.svg?branch=master)

A small package for optimizing banana coverage on peanut butter and banana sandwiches.


![assets/perfect_sandwich.jpg](assets/perfect_sandwich.jpg)


## Installation

Unfortunately, one of the dependencies, [nest2D](https://github.com/markfink/nest2D), is a pain to install (it has a bunch of C dependencies), so I am unable to publish this package to PyPi right now. I have an open [PR](https://github.com/markfink/nest2D/pull/2) to hopefully solve some of the C issues, and then I will publish to PyPi.

In the meantime, you can install the package locally.
 
 First, install [boost](https://www.boost.org/) and [cmake](https://cmake.org/). I installed them on my Mac as follows:

```commandline
conda install cmake
brew install boost
```

After that, git clone this repo and use [poetry](https://python-poetry.org/docs/) to install the package.

```commandline
poetry install
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
