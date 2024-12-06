import React, {Component} from "react";
import {hierarchyScale} from "../scales/hierarchyScale";
import * as d3 from "d3"
import {divergingDistributionColorScale} from "../scales/divergingDistributionColorScale";
import * as vsup from "vsup";

export default class HorizontalIciclePlot extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            parentG: null,
            positionalHierarchyData: null,
            size: null,
            paddingInner: 0.01,
            paddingOuter: 0.01,
            experimentNames: null,
            timestamp: 0,
            hiftPressed: false, // Track Shift key state
        }

        var filter = d3.select(this.gref.current).append("defs")
            .append("filter")
            .attr("id", "shadow")
            .attr("x", "-5%")
            .attr("y", "-5%")
            .attr("width", "250%")
            .attr("height", "140%")

        filter.append("feOffset")
            .attr("result", "offOut")
            .attr("in", "SourceAlpha")
            .attr("dx", "0")
            .attr("dy", "10");

        filter.append("feGaussianBlur")
            .attr("result", "blurOut")
            .attr("in", "offOut")
            .attr("stdDeviation", "5");

        filter.append("feComponentTransfer")
            .attr('result', 'opacityAdjusted')
            .attr("in", "blurOut")
            .attr("type", "linear")
            .attr("slope", "0.2"); // Adjust this value between 0 and 1 to control the opacity

        filter.append("feBlend")
            .attr("in", "SourceGraphic")
            .attr('in2', 'blurOut')
            .attr("in3", "opacityAdjusted")
            .attr("mode", "normal");

        // Action binding
        this.brushAction = this.brushAction.bind(this)
        this.brushActionDuring = this.brushActionDuring.bind(this)
        this.clickAction = this.clickAction.bind(this)
        this.clickBarAction = this.clickBarAction.bind(this)
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {parentG, positionalHierarchyData, colorScale,
            // avgMagnitude, avgStd, m_std_interval, u_std_interval,
            size, translate, visDepth, experimentNames} = nextProps;

        if (!positionalHierarchyData)
            return null

        // if (!hierarchyData)
        //     return null

        if (!colorScale)
            return null

        // Skewed scaleBand
        let rangeSchedule = [10, 50, 150, 200, 500]
        let depthScale = d3.scaleBand()
            .domain(d3.range(visDepth + 1))
            .range([0, size[1]])
            // .range(rangeSchedule)

        // Value
        // let spacing = 1
        // let squareQuantization = vsup.squareQuantization().n(4)
        //     // .valueDomain([avgMagnitude - (2 * u_std_interval), avgMagnitude + (2 * u_std_interval)])
        //     .valueDomain([avgMagnitude + (spacing * u_std_interval), avgMagnitude - (spacing * u_std_interval)])
        //     .uncertaintyDomain([avgStd - (spacing * m_std_interval), avgStd + (spacing * m_std_interval)]);
        // var colorScale = vsup.scale().quantize(squareQuantization).range(d3.interpolateViridis);

        return {parentG, positionalHierarchyData, depthScale, colorScale, size, translate, visDepth, experimentNames}
    }

    drawPlot() {
        /*
        To limit nodes to certain sizes, I can take two approaches.

          1. Modify the data - merge small nodes into a leaf node
          2. Selectively show nodes in the visualization.
         */
        console.log("Drawing horizontal plot")

        const visDepth = this.props.visDepth
        // const size = this.props.size

        let nodeG = d3.select(this.gref.current).selectAll("g")
            .data([this.props.positionalHierarchyData], d => d.name) // Supply a key to match the data.
            .join('g')
            .attr('transform', d => `translate(${d.position}, 0)`)
            .classed('iciclenode', true)
        let rootNodeG = d3.select(this.gref.current).selectAll("g")

        nodeG.exit().remove()

        for(let i = 0; i < visDepth; i++) {
            if (i <= visDepth) { // Not the leaf nodes.
                nodeG = nodeG.selectAll()
                    .data(d => d.children, d => d.name, d => d.flatIdx)
                    .join('g')
                    .attr('transform', d => {
                        return `translate(${d.position}, 0)`
                    })
                    .classed('iciclenode', true)
                    .classed('horizontalnode', true)
            } else {
                // nodeG = nodeG.selectAll()
                //     .data(d => d.children, d => d.name)
                //     .join('g')
                //     .classed('pseudoiciclenode', true)
            }

            let exit = nodeG.exit()
            exit.remove()
        }
        // Just variance
        // const varianceScale = d3.scaleLinear().domain([this.props.stdFloor, (this.props.stdCeil + this.props.stdFloor) / 2, this.props.stdCeil]).range(['blue', 'white', 'red'])
        // const varianceScale = d3.scaleLinear().domain([1.5, 2, 2.5]).range(['blue', 'white', 'red'])
        // console.log([this.props.stdFloor, this.props.stdCeil])

        // Give IDs
        // d3.select(this.gref.current).selectAll('.iciclenode')
        const S = 0.15
        d3.select(this.gref.current).selectAll('.iciclenode')
            .append('rect')
                .attr('x', 0)
                .attr('y', (d, i) => {
                    // return this.state.depthScale(d.depth)

                    // Less height for non-leaf nodes. Leaf node to be fixed at {S}% of all available height, other
                    // nodes share the remainder. The y position need to be adjusted.
                    const Sx = (1 - S) / (this.props.visDepth - 1)
                    const totDepth = this.state.depthScale.bandwidth() * this.props.visDepth // Total visDepth
                    const shift = (totDepth * Sx) * d.depth
                    return shift
                })
                .attr('width', d => d.size)
                .attr('height', d => {
                    // Less height for non-leaf nodes. Leaf node to be fixed at {S}% of all available height, other
                    // nodes share the remainder.
                    // return this.state.depthScale.bandwidth()

                    // Total visDepth
                    const totDepth = this.state.depthScale.bandwidth() * this.props.visDepth
                    if (d.depth === this.props.visDepth) {
                        return totDepth * S
                    }
                    return totDepth * ((1 - S) / (this.props.visDepth - 1))
                })
                .attr('stroke', 'black')
                .attr('stroke-width', 0.75) // This would do nothing.
                .attr('fill', d => {
                    return this.state.colorScale(d.magnitude, d.var)
                    // console.log(this.props.stdFloor, d.var, this.props.stdCeil)
                    // return varianceScale(d.var)
                })
                .attr('rx', 5)
                .on('click', this.clickAction)
                .lower()

        // Visual retention
        this.nodeHighlighting(this.props.getSelectionDirection().map(d => d.name))
    }

    removeItemOnce(arr, value) {
        var index = arr.indexOf(value);
        if (index > -1) {
            arr.splice(index, 1);
        }
        return arr;
    }

    // nodeHighlighting(enabledSelection) {
    //     d3.select(this.gref.current).selectAll('.iciclenode').select('rect')
    //         .attr('stroke', 'black')
    //         .style('filter', null)
    //         .attr('stroke', d3.hcl(0, 0, 30))
    //         .attr('stroke-width', 0.75)
    //         .raise()
    //         // .attr('stroke-width', d => {
    //         //     return 0.75 + (d.weaveScore * 3)
    //         // })
    //
    //     // Re-clustering (membership change)
    //     d3.select(this.gref.current).selectAll('.iciclenode').filter(d => d.depth === this.props.visDepth)
    //         .append('circle')
    //         .attr('cx', d => d.size / 2)
    //         .attr('cy', this.state.size[1] - 5.5)
    //         // .attr('cy', this.state.size[1] - 23.5)
    //         .attr('r', d => {
    //             const r = ((d.weaveScore * 4) / this.props.weaveMax)
    //             return isNaN(r) ? 0 : r
    //         })
    //         .style('fill', d3.hcl(360, 100, 70))
    //
    //     d3.select(this.gref.current).selectAll('.iciclenode').select('rect')
    //         .filter(d => enabledSelection.includes(d.name))
    //         .filter(d => d.depth === this.state.visDepth)
    //         .style('filter', 'url(#shadow)')
    //         // .attr('stroke', d3.hcl(120, 90, 40))
    //         .attr('stroke', d3.hcl(50, 100, 75))
    //         .attr('stroke-width', 4)
    //         .raise()
    //         // .attr('stroke', 'red')
    // }

    nodeHighlighting(enabledSelection) {
        const gref = d3.select(this.gref.current);

        // Reset styles for all nodes
        gref.selectAll('.iciclenode').select('rect')
            .attr('stroke', d3.hcl(0, 0, 30))
            .attr('stroke-width', 0.75)
            .style('filter', null);

        // Highlight selected nodes
        gref.selectAll('.iciclenode')
            .filter(d => enabledSelection.includes(d.name))
            // .filter(d => enabledSelection.includes(d.name) && d.depth === this.state.visDepth)
            .each(function () {
                d3.select(this).raise(); // Raise the entire group
            })
            .select('rect')
            .style('filter', 'url(#shadow)')
            .attr('stroke', d3.hcl(50, 100, 75))
            .attr('stroke-width', 4);

        // Add circles for re-clustering nodes
        gref.selectAll('.iciclenode')
            // .filter(d => d.depth === this.props.visDepth)
            .append('circle')
            .attr('cx', d => d.size / 2)
            .attr('cy', this.state.size[1] - 5.5)
            .attr('r', d => {
                const r = ((d.weaveScore * 4) / this.props.weaveMax);
                return isNaN(r) ? 0 : r;
            })
            .style('fill', d3.hcl(360, 100, 70));
    }


    getUpdatedSelectionDirections(selection) {
        let prevSelectionDirection = this.props.getSelectionDirection()
        let prevSelectionDirectionNames = prevSelectionDirection.map(d => d.name)

        if (prevSelectionDirectionNames.includes(selection.name)) {
            const idx = prevSelectionDirection.map(d => d.name).indexOf(selection.name)
            prevSelectionDirection.splice(idx, 1)
        }
        else
            prevSelectionDirection.push(selection)
        return prevSelectionDirection
    }

    clickAction(e, d) {
        // Update data and Visually update - For this I need to know which nodes are selected.
        // This information lives in the parent component.
        // I need a function to retrieve that information from parent component.
        let prevSelectionDirection = this.getUpdatedSelectionDirections(d)
        let prevSelectionDirectionNames = prevSelectionDirection.map(d => d.name)

        // Visual update
        this.nodeHighlighting(prevSelectionDirectionNames)

        this.props.clickListener(prevSelectionDirection)  // Pass it to listener for data processing
    }

    clickBarAction(e, d) {
        // let prevSelectionDirection = this.getUpdatedSelectionDirections(d)
        // let prevSelectionDirectionNames = prevSelectionDirection.map(d => d.name)

        this.props.clickBarListener(d.name)
    }

    brushActionDuring(e, d) {
        let brushSelection = e.selection
        let brushedItems = []

        if (!brushSelection)  // Prevents error when brush is empty.
            return

        // Visually
        d3.select(this.gref.current).selectAll('.iciclenode')
            .filter(d => d.depth === this.state.visDepth)
            .filter(function(d) {
                let start = d.absolute_position
                let end = start+d.size-0.001
                if(brushSelection[1] < start || end < brushSelection[0])
                    return false

                brushedItems.push(d)
                return true
            })

        // Visual update
        this.nodeHighlighting(brushedItems.map(d => d.name))

        this.props.brushListenerDuring(brushedItems, brushSelection[0], brushSelection[1])
    }

    brushAction(e, d) {
        let brushSelection = e.selection
        let brushedItems = []

        if (!brushSelection)  // Prevents error when brush is empty.
            return

        // Visually
        d3.select(this.gref.current).selectAll('.iciclenode')
            .filter(d => d.depth === this.state.visDepth)
            .filter(function(d) {
                let start = d.absolute_position
                let end = start+d.size-0.001
                if(brushSelection[1] < start || end < brushSelection[0])
                    return false

                brushedItems.push(d)
                return true
            })

        // Visual update
        this.nodeHighlighting(brushedItems.map(d => d.name))

        this.props.brushListener(brushedItems, brushSelection[0], brushSelection[1])
    }

    hoverAction(e, d) {
        let std = d.data().var

    }

    drawBrush() {
        // Brush is drawn once during initialization, but never unattached afterward. Control this in componentDidUpd.
        let brush = d3.brushX().extent([[0,0],[this.state.size[0], this.state.depthScale.bandwidth()]])
        brush.on('brush', this.brushActionDuring)
        brush.on('end', this.brushAction)

        d3.select(this.state.parentG).append('g')
            .attr('transform', `translate(${this.state.translate[0]}, 
                                          ${this.state.depthScale(this.state.visDepth) + this.state.translate[1]})`)
            // .attr('z-index', 999)
            .classed('brush', true)
            .classed('horizontal_brush', true)
            .call(brush)
            .raise()
    }

    deleteBrush() {
        d3.select(this.state.parentG).selectAll('.horizontal_brush').remove()
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        /*
        Note that this function usually simply returns true or false - to indicate if the "render" function should be
        invoked. However, I am manually updating this component per condition. This allows me to control inidividual
        aspect of the d3 component.

         */
        // console.log("update horizontal icicle plot")

        if (prevProps.positionalHierarchyData == null && this.props.positionalHierarchyData == null) {
            return;
        }

        if (prevProps.positionalHierarchyData == null && this.props.positionalHierarchyData) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return;
        }

        if (prevProps.positionalHierarchyData.timestamp !== this.props.positionalHierarchyData.timestamp) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return;
        }

        if (prevState.visDepth !== this.state.visDepth) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return
        }

        if (prevState.experimentNames !== this.state.experimentNames) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return
        }

        // position hierarchy data's timestamps change on re-request of the base data.
        // if (prevState.positionalHierarchyData.timestamp !== this.state.positionalHierarchyData.timestamp) {
        //     this.drawPlot()
        // }

        if (prevState.colorScale !== this.state.colorScale) {
            this.drawPlot()
        }
    }

    render() {
        if (!this.state.translate)
            return

        return (
            <g ref={this.gref}
               transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`}
               width={this.props.size[0]}
               height={this.props.size[1]}/>
        )
    }
}