import { Component, createRef } from "react";
import App from "./App";
import $ from 'jquery';

// Style
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import * as d3 from "d3";

const EXPERIMENTS = {
    "CelebA 256": [
        {id: "ldm_celeba256-vac-global-all", name: "LDM CelebA VAC Global"},
        {id: "s2_celeba256-vac-global-early_0", name: "CelebA VAC Global Early 0"},
        {id: "s2_celeba256-sefakmc-global-early_0", name: "CelebA SeFA Global Early 0"},
        {id: "s2_celeba256-sefakmc-global-early_1", name: "CelebA SeFA Global Early 1"}
    ],
    "FFHQ 256": [
        {}
    ]
};


export default class AppUpper extends Component {
    constructor() {
        super();
        this.svgRef = createRef();
        this.state = {
            availableExperiments: [],
            experimentNames: [],
            methodBlindMode: false,
            processingMethod: 'end',
            clusteringMethod: 'complete',
            pairwiseMetric: 'cosine',

            // Vis
            // height: 960, // 640
            height: 1080, // 640
            // width: 1200,
            width: 1440,
            // icicleSize: 125,
            icicleSize: 160,
            settingWidth: 0,
            toggledBarSize: 8,
            originalImagePlotSize: 120,
            imageSize: 100,
            oriGap: 10,
            visDepth: 7,
            truncatedTree: true,

            // Infra
            port: 37203
        };

        this.requestAvailableExperiments = this.requestAvailableExperiments.bind(this);
        this.requestAvailableExperiments();
        this.checkboxOnChange = this.checkboxOnChange.bind(this);
        this.methodBlindModeOnChange = this.methodBlindModeOnChange.bind(this);
        this.processingMethodOnChange = this.processingMethodOnChange.bind(this);
        this.clusteringMethodOnChange = this.clusteringMethodOnChange.bind(this);
        this.pairwiseMetricOnChange = this.pairwiseMetricOnChange.bind(this);
        this.visDepthOnChange = this.visDepthOnChange.bind(this);
        this.truncatedTreeOnChange = this.truncatedTreeOnChange.bind(this);
    }

    checkboxOnChange(e) {
        let newExperimentNames = [...this.state.experimentNames];
        if (newExperimentNames.includes(e.target.value))
            this.removeItemOnce(newExperimentNames, e.target.value);
        else
            newExperimentNames.push(e.target.value);

        this.setState({ experimentNames: newExperimentNames });
    }

    async requestAvailableExperiments() {
        try {
            let response = await fetch(`http://127.0.0.1:${this.state.port}/conceptlens/available_experiments`, {
                method: 'POST',
                body: JSON.stringify({})
            });
            let data = await response.json();
            this.setState({ availableExperiments: data });
        } catch (error) {
            console.error('Error fetching experiments:', error);
        }
    }

    removeItemOnce(arr, value) {
        var index = arr.indexOf(value);
        if (index > -1) {
            arr.splice(index, 1);
        }
        return arr;
    }

    methodBlindModeOnChange(e) { this.setState({ methodBlindMode: e.target.value }) }
    processingMethodOnChange(e) { this.setState({ processingMethod: e.target.value }) }
    clusteringMethodOnChange(e) { this.setState({ clusteringMethod: e.target.value }) }
    pairwiseMetricOnChange(e) { this.setState({ pairwiseMetric: e.target.value }) }
    visDepthOnChange(e) { this.setState({ visDepth: parseInt(e.target.value) }) }
    truncatedTreeOnChange(e) { this.setState({ truncatedTree: e.target.value }) }

    renderExperimentCheckboxes() {
        return Object.keys(EXPERIMENTS).map(section => (
            <div key={section}>
                <h5>{section}</h5>
                {EXPERIMENTS[section].map(exp => (
                    <Row key={exp.id}>
                        <Col xs={9}>{exp.name}</Col>
                        <Col>
                            <input type="checkbox" value={exp.id} onChange={this.checkboxOnChange} />
                        </Col>
                    </Row>
                ))}
            </div>
        ));
    }

    render() {
        let toggledBarHeight = 8;
        return (
            <Container fluid>
                <br /><br />
                <Row>
                    <Col xs={2}>
                        <Row className={'row'}>
                            <h4> Methods </h4>
                            {this.renderExperimentCheckboxes()}
                        </Row>

                        <hr />

                        <Row onChange={this.processingMethodOnChange}>
                            <h5> Feature Processing Method Selection </h5>
                            <Row>
                                <Col> Vector Difference </Col>
                                <Col>
                                    <input type="radio" value="diff" name="pmd" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Vector End </Col>
                                <Col>
                                    <input type="radio" value="end" name="pmd" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.clusteringMethodOnChange}>
                            <h5> Clustering Method </h5>
                            <Row>
                                <Col> Complete </Col>
                                <Col>
                                    <input type="radio" value="complete" name="cm" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Ward </Col>
                                <Col>
                                    <input type="radio" value="ward" name="cm" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.pairwiseMetricOnChange}>
                            <h5> Pairwise Distance Metric </h5>
                            <Row>
                                <Col> Cosine </Col>
                                <Col>
                                    <input type="radio" value="cosine" name="pdm" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Euclidean </Col>
                                <Col>
                                    <input type="radio" value="euclidean" name="pdm" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Raw </Col>
                                <Col>
                                    <input type="radio" value="raw" name="pdm" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.truncatedTreeOnChange}>
                            <h5> Tree Truncation </h5>
                            <Row>
                                <Col> True </Col>
                                <Col>
                                    <input type="radio" value="True" name="tt" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> False </Col>
                                <Col>
                                    <input type="radio" value="False" name="tt" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.visDepthOnChange}>
                            <h5> Tree Visualization Depth </h5>
                            <Row>
                                <Col>
                                    <input type="range" min="1" max="12" step="1" className="slider" />
                                </Col>
                            </Row>
                        </Row>
                    </Col>
                    <Col lg={10}>
                        <App
                            experimentNames={this.state.experimentNames}
                            methodBlindMode={this.state.methodBlindMode}
                            featureProcessingMethod={this.state.processingMethod}
                            clusteringMethod={this.state.clusteringMethod}
                            pairwiseMetric={this.state.pairwiseMetric}
                            truncatedTree={this.state.truncatedTree}
                            visDepth={this.state.visDepth}
                            height={this.state.height}
                            width={this.state.width}
                            icicleSize={this.state.icicleSize}
                            originalImagePlotSize={this.state.originalImagePlotSize}
                            imageSize={this.state.imageSize}
                            toggledBarHeight={toggledBarHeight}
                        />
                    </Col>
                </Row>
            </Container>
        );
    }
}
