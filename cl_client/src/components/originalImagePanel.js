import React, {Component} from "react";
import {biTreeColorScale} from "../scales/biTreeColorScale"
import * as d3 from "d3"
import {accumulateLeafNodesBudget, accumulateVisLeafNodes} from "../helper_functions/accumulateLeafNodes";


function evenlySampleArray(arr, n) {
    return Array.from({length: n}, (_, i) => arr[Math.floor(i * arr.length / n)]);
}

export default class OriginalImagePanel  extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            parentG: null,
            size: null,
            paddingInner: 0.01,
            paddingOuter: 0.01,
            width: 125,
            methodColorScale: null
        }

        this.joinImagePanel = this.joinImagePanel.bind(this)
        this.setupCodeImages = this.setupCodeImages.bind(this)
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {parentG, positionalHierarchyCode, size, translate, visDepth} = nextProps;

        return {parentG, positionalHierarchyCode, size, translate, visDepth};
    }

    joinImagePanel = () => {
        let biColorScale = biTreeColorScale([0,10], [40,97],4)

        let setup_single_rects = (selection, drawLine) => {
            selection
                .attr('width', this.state.width)
                .attr('height', d => d.size)
                .attr('fill', 'none')
                .attr('rx', 5)

            if (!drawLine) {
                selection.attr('stroke', 'none')
            } else {
                selection.attr('stroke', d3.hcl(0,0,30))
                .attr('fill', d => {
                    let max_d = Math.max(d.depth)
                    return 'none'
                })
            }

        }

        let enter_ops = (enter, drawLine) => {
            let g = enter.append('g')
                .attr('transform', d => `translate(0, ${d.position})`)
                .classed('rnode', true)

            g.append('rect')
                .call(setup_single_rects, drawLine)

            g.append('g')
                .filter(d => d.depth === this.state.visDepth)
                .call(this.setupCodeImages)
        }

        let update_ops = (update, drawLine) => {
            update
                .attr('transform', d => `translate(${0}, ${d.position})`)
                .select('rect') // This propagates data attached to group element to a child element, which is 'rect' element.
                .call(setup_single_rects, drawLine)
        }

        d3.select(this.gref.current).selectAll('.rnode')
            .data([this.state.positionalHierarchyCode], d => 'r-' + d.name)
            .join(
                enter => enter_ops(enter, true),
                update => update_ops(update, true),
                exit => exit.remove()
            )

        for(let i = 0; i < this.state.visDepth; i++) {
            let next_depth_g = d3.select(this.gref.current).selectAll('.rnode').filter(d => d.depth === i)
            next_depth_g.selectAll('.rnode')
                .data(d => d.children, d => 'r-' + d.name)
                .join(
                    enter => enter_ops(enter, false),
                    update => update_ops(update, false),
                    exit => exit.remove()
                )
        }
    }

    setupCodeImages(selection) {
        let methodColorScale = d3.scaleSequential(d3.schemePastel1).domain([0, 1])

        let getImageLink = (codeIdx, flatIdx, treeID) => {
            let bucketPath = `http://localhost:${this.props.port}/served_data/${this.props.experimentNames[0]}/`
            return bucketPath + 'codes/' + `${codeIdx}.jpg`
        }

        let insertImage = (selection, imageLink, imageSize, xPos, yPos, aorb) => {
            // Sets up single image
            selection
                .attr('onerror', "this.style.display='none'")
                .attr("xlink:href", imageLink)
                .attr('transform', `translate(${xPos}, ${yPos})`)
                .attr('width', d => imageSize)
                .attr('height', d => imageSize)
        }

        let topG = selection.filter(d => (d.depth === this.state.visDepth)) // Draw only for the top boxes.
        topG
            .attr('width', this.state.width)
            .attr('height', d => d.size)

        // Number of topGs
        let topGcount = selection.size()

        // scales for individual boxes
        topG.each(function(d, i) {
            const width = 125
            const height = d.size
            const selection = d3.select(this)
            let nodeData = selection.data()[0]

            let imageSize = 120

            // Determine how many images can fit into each box.
            let horizontalCount = 1
            let verticalCount = Math.floor(height / imageSize)  // Code budget

            // Use width and height to determine padding.
            const horizontalPadding = (width - (imageSize * horizontalCount)) / 2
            const verticalPadding = (height - (imageSize * verticalCount)) / 2

            const horizontalScale = d3.scaleBand().domain(Array.from({length: horizontalCount}, (_, i) => i))
                .range([horizontalPadding, width - horizontalPadding]).paddingOuter(0.00).paddingInner(0.1)

            const verticalScale = d3.scaleBand().domain(Array.from({length: verticalCount}, (_, i) => i))
                .range([verticalPadding, height - verticalPadding]).paddingOuter(0.00).paddingInner(0.1)

            // Leaf node indices
            let codeLeaf = accumulateVisLeafNodes(nodeData)

            if (codeLeaf.length < verticalCount) {
                verticalCount = codeLeaf.length
            }

            let codeSample = evenlySampleArray(codeLeaf, verticalCount)

            for(var v = 0; v < verticalCount; v++) {
                const imageLink = getImageLink(codeSample[v].name, 0, 0)
                if (codeSample[v]) {
                    selection
                        .append('image')
                        .call(insertImage, imageLink, verticalScale.bandwidth(), horizontalScale(0), verticalScale(v))
                }
            }
        })
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevState.positionalHierarchyCode !== this.state.positionalHierarchyCode) {
            this.joinImagePanel()
        }
    }

    render() {
        if(!this.state.translate)
            return

        return (
            <g ref={this.gref}
               transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`}
               width={this.state.size[0]}
               height={this.state.size[1]}/>
        )
    }
}