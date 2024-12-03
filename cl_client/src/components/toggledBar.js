import React, {Component} from "react";
import * as d3 from "d3"
import {accumulateLeafNodesBudget} from "../helper_functions/accumulateLeafNodes";
import {splitExperimentName} from "../helper_functions/splitExperimentName";


export default class ToggledBar extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            leafNodes: []
        }
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {positionalHierarchyDirection, methodColorScale, size, translate} = nextProps;

        let leafNodes = null
        try {
            leafNodes = accumulateLeafNodesBudget(positionalHierarchyDirection)
        } catch (e) {

        }

        return {positionalHierarchyDirection, leafNodes, size, translate, methodColorScale};
    }

    drawComponent() {
        if (!this.state.leafNodes)
            return

        // d3.select(this.gref.current).selectAll('rect').clear()

        d3.select(this.gref.current).selectAll()
            .data(this.state.leafNodes)
            .join('rect')
            .attr('width', (this.state.size[0] / this.state.leafNodes.length) / 1.0)
            .attr('height', this.state.size[1])
            .attr('x', (d, i) => {
                return (this.state.size[0] / this.state.leafNodes.length) * i
            })
            .attr('y', 0)
            .attr('fill', (d, i) => {
                let [domainName, methodName, applicationName, layerName, layerSubName] = splitExperimentName(d.expName)
                methodName = methodName + ' ' + applicationName

                let color = d3.hcl(
                    this.props.methodColorScale.hue(methodName),
                    this.props.methodColorScale.chr(methodName),
                    this.props.methodColorScale.lum(methodName) - this.props.methodColorScale.lay(layerName))

                // let color = d3.hcl(
                //     this.props.methodColorScale.hue(largest),
                //     this.props.methodColorScale.chr(largest),
                //     this.props.methodColorScale.lum(largest) - this.props.methodColorScale.lay(layerName))

                return color

            })
            // .attr('opacity', 0.92)
            // .style('mix-blend-mode', 'multiply')

    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevState.positionalHierarchyDirection !== this.state.positionalHierarchyDirection)
            return true
    }

    render() {
        this.drawComponent()
        return (
            <g ref={this.gref} transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`} onClick={this.props.onclick}/>
        )
    }
}