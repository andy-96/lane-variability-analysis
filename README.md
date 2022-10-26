# ⚠️ Code is currently in a review process and will be uploaded soon

# Lane Variability Analysis - A learning-based approach

## Motivation and Purpose

*TL;DR: Differentiate between automation levels and automation systems of vehicles by analyzing the lane variability, i.e. the variability of the lateral position*

In a recent report, it was forecasted that in 2015, over 59\% of total vehicles sold will have automation level 2 capabilities. Vehicles with level 2 capabilities, i.e. partial automation systems, are equipped with ADAS that can brake, accelerate, and steer, such as lane centering systems or adaptive cruise control which are allowed to be used on highways only. With the growing number of partially automated vehicles on the roads, we must understand how and why drivers use these systems and especially, when is a system failing, i.e. the driver interferes to further improve the systems in a human-centric way. Currently, the only way to determine if a vehicle is driven manually or partially automated is to decode the CAN-Bus signal which is time-consuming and has to be done for every single unique vehicle model.

This model's purpose is to distinguish whether a vehicle is driven manually or partially automated by only using the extracted lane markings.

## Architecture
![picture of architecture](assets/lva-architecture.jpg)

Inspired by SimCLR, a contrastive loss is used to compare two samples with each other attempting to distinguish whether they are similar or dissimilar. Hence, to provide a supervisory signal to the model, labels on whether the sample pair are in fact similar, i.e. positive, or dissimilar, i.e. negative, are needed. Since the automation level and vehicle model labels are easily obtained via the CAN-bus and the study setup, they are leveraged to actively sample for positive and negative samples.

As seen in the architecture, the time-series encoder ResCNN is used for encoding the trajectories. Emperically, I have observed that encoding the image of a trajectory plot improves the model's performance a lot, which is why a ResNet-18 is used to encode the plots in parallel to the ResCNN. Given the stacked feature maps of the image and time series encoder, another neural network, i.e. the projection head, is used to reduce dimensionality to learn even stronger representations. The contrastive loss and triplet loss respectively is then applied on the output of the projection head. Furthermore, a classification head with a few fully connected layers and a mean squared error (MSE) is used to classify the input data.

## How to run

### Setup

Create and start the conda environment by running

```
conda create --name lva --file pkgs.txt
conda activate lva
```

### Data Prep
Before running the model, the lateral position, i.e. trajectories, needs to be filtered from the previously extracted lane detections. For this, please use the following jupyter notebook: `notebooks/filter_lanes.ipynb`

### Train model

All models are saved under `./experiments`. To train model, run `./scripts/sample.sh`.

#### Important Arguments

`--comment` \<datetime>_\<comment>_\<timeseries-encoder>_\<car model>_\<automation level>_\<batch size>

`--data_mode` Which encoders to use: timeseries encoder only, image only or both

**Loss-related**

`--loss_setup` which loss to use when and weighted by what. The format is the following: <loss-function> <start_epoch> <loss-function> <start_epoch> ...

`--k 100` Temperature coefficient for contrastive loss

`--loss_margin` Margin for triplet loss

**Data-related**

`--automation_level` Use L0 and/or L2 data. E.g. if only set as 0, all level 2 data is filtered out

`--car_model` Similar as automation level but with car model

`--classes` By which the model should try to separate the data. E.g. if automation level is set, the model tries to differentiate by automation level

### Infer/Analyze model

All plots and analysis related plots are in `notebooks/inference.ipynb`, i.e. TSNE visualization, confusion matrix, inertia/Dunn-Index and accuracy.

## Results

**1. Case: Cadillac CT6 - Level 0 vs. Level 2**

<table>
    <thead>
        <tr>
            <th rowspan=2>Vehicle</th>
            <th rowspan=2>Autom. Level</th>
            <th colspan=3 style="text-align:center">Train</th>
            <th colspan=3 style="text-align:center">Test</th>
        </tr>
        <tr>
            <th>Inertia &#8595;</th>
            <th>Dunn &#8593;</th>
            <th>Acc &#8593;</th>
            <th>Inertia &#8595;</th>
            <th>Dunn &#8593;</th>
            <th>Acc &#8593;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Cadillac</td>
            <td>Level 0</td>
            <td>0.240</td>
            <td>0.819</td>
            <td rowspan=2>100.0%</td>
            <td>0.171</td>
            <td>1.195</td>
            <td rowspan=2>85.1%</td>
        </tr>
        <tr>
            <td>Cadillac</td>
            <td>Level 2</td>
            <td>0.059</td>
            <td>3.734</td>
            <td>0.128</td>
            <td>1.109</td>
        </tr>
    </tbody>
</table>

**2. Case: Tesla Model S/X - Level 0 vs. Level 2**


<table>
    <thead>
        <tr>
            <th rowspan=2>Vehicle</th>
            <th rowspan=2>Autom. Level</th>
            <th colspan=3 style="text-align:center">Train</th>
            <th colspan=3 style="text-align:center">Test</th>
        </tr>
        <tr>
            <th>Inertia &#8595;</th>
            <th>Dunn &#8593;</th>
            <th>Acc &#8593;</th>
            <th>Inertia &#8595;</th>
            <th>Dunn &#8593;</th>
            <th>Acc &#8593;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Tesla</td>
            <td>Level 0</td>
            <td>0.150</td>
            <td>0.457</td>
            <td rowspan=2>97.6%</td>
            <td>0.145</td>
            <td>1.238</td>
            <td rowspan=2>82.4%</td>
        </tr>
        <tr>
            <td>Tesla</td>
            <td>Level 2</td>
            <td>0.062</td>
            <td>1.351</td>
            <td>0.092</td>
            <td>1.357</td>
        </tr>
    </tbody>
</table>

**3. Case: Level 2 - Cadillac CT6 vs. Tesla Model S/X**


<table>
    <thead>
        <tr>
            <th rowspan=2>Vehicle</th>
            <th rowspan=2>Autom. Level</th>
            <th colspan=3 style="text-align:center">Train</th>
            <th colspan=3 style="text-align:center">Test</th>
        </tr>
        <tr>
            <th>Inertia &#8595;</th>
            <th>Dunn &#8593;</th>
            <th>Acc &#8593;</th>
            <th>Inertia &#8595;</th>
            <th>Dunn &#8593;</th>
            <th>Acc &#8593;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Cadillac</td>
            <td>Level 2</td>
            <td>0.134</td>
            <td>0.582</td>
            <td rowspan=2>95.3%</td>
            <td>0.160</td>
            <td>0.750</td>
            <td rowspan=2>91.9%</td>
        </tr>
        <tr>
            <td>Tesla</td>
            <td>Level 2</td>
            <td>0.059</td>
            <td>1.609</td>
            <td>0.081</td>
            <td>1.868</td>
        </tr>
    </tbody>
</table>

**4. Case: Cadillac CT6 vs. Tesla Model S/X vs. Level 0 vs. Level 2**

## Pretrained models

- For all three models!
