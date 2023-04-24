# FlowMapVideo

FlowMapVideo is a tool for visualizing the evolution of traffic flow over time. The output is a video with a map showing the traffic flow intensities at each time interval. The animation is generated using the [FlowMapFrame](https://github.com/It4innovations/FlowMapFrame) library to render individual frames based on the output of the [Ruth](https://github.com/It4innovations/ruth) traffic simulator.


https://user-images.githubusercontent.com/95043942/234028881-b3f9298b-1aa8-483e-bddd-3b8685f37bcf.mp4


## Installation

### Prerequisites

To run, you need to install `FFmpeg` and [Ruth](https://github.com/It4innovations/ruth).

##

1. Create and activate a virtual environment:
```
virtualenv <VENV>
source <VENV>/bin/activate
```


2. Install via pip
```
python3 -m pip install git+https://github.com/It4innovations/FlowMapVideo.git
```

## Run
```
traffic-flow-map --help
```

