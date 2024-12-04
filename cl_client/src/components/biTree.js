import React, {Component} from "react";
import {biTreeColorScale} from "../scales/biTreeColorScale"
import * as d3 from "d3"
import {accumulateLeafNodesBudget, accumulateVisLeafNodes} from "../helper_functions/accumulateLeafNodes";
import {splitExperimentName} from "../helper_functions/splitExperimentName";


function evenlySampleArray(arr, n) {
    return Array.from({length: n}, (_, i) => arr[Math.floor(i * arr.length / n)]);
}

export default class BiTree extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            parentG: null,
            size: null,
            imageSize: 100,
            paddingInner: 0.01,
            paddingOuter: 0.01,
            methodColorScale: [],
            contributions: [],
        }

        this.joinBiTree = this.joinBiTree.bind(this)
        this.setup_rects = this.setup_rects.bind(this)
        this.enter_ops = this.enter_ops.bind(this)
        this.update_ops = this.update_ops.bind(this)
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {
            parentG,
            positionalHierarchyCode,
            positionalHierarchyDirection,
            methodColorScale,
            size,
            imageSize,
            translate,
            visDepth,
            contributions,
            magmin,
            magmax
        } = nextProps;


        // Calculate the extent for both magnitude and variance
        const magExtent = d3.extent(contributions, d => d.mag_contribution);
        const varExtent = d3.extent(contributions, d => d.var_contribution);

        // Define scales based on the computed extents
        const magScale = d3.scaleLinear()
            .domain(magExtent)
            // .domain([10, 12])
            .range([0, 1]);  // Scale from 0 to 1 (to adjust for the full height of the bar)

        const varScale = d3.scaleLinear()
            .domain(varExtent)
            // .domain([0.05, 0.25])
            .range([0, 1]);  // Scale from 0 to 1 (to adjust for the full width of the bar)

        const codeGroupedMagnitude = d3.groups(contributions, d => d.code)
        const directionGroupedMagnitude = d3.groups(contributions, d => d.direction)
        // console.log("code", codeGroupedMagnitude)
        // console.log(directionGroupedMagnitude)

        return {
            parentG,
            positionalHierarchyCode,
            positionalHierarchyDirection,
            methodColorScale,
            size,
            imageSize,
            translate,
            visDepth,
            contributions,
            magScale,
            varScale,
            codeGroupedMagnitude,
            directionGroupedMagnitude,
            magmin,
            magmax
        };
    }

    setup_rects = (selection) => {
        selection
            .attr('width', d => {
                return d[0].size
            })
            .attr('height', d => d[1].size)
            .attr('stroke', d3.hcl(0, 0, 30))
            .attr('fill', 'none')
            .attr('rx', 5)
            .transition()
    }

    enter_ops(enter) {
        let g = enter.append('g')
            .attr('transform', d => {
                // console.log("Positions: ", d[0].depth, d[1].depth, d[0].position, d[1].position, d[0], d[1])
                return `translate(${d[0].position},${d[1].position})`
            })
            .classed('node', true)

        // Rectangles
        g.append('rect').call(this.setup_rects)
        let selection = g.append("g").filter(d =>
            d[0].depth === this.state.visDepth &&
            d[1].depth === this.state.visDepth) // Nested Images

        const allPositions = selection.data().map(d => ({
            rect: d,
            x: d[0].absolute_position,
            y: d[1].absolute_position,
        }));

        const minY = d3.min(allPositions, d => d.y);
        const minX = d3.min(allPositions, d => d.x)
        const topRowRects = allPositions.filter(d => d.y === minY);
        const leftColRects = allPositions.filter(d => d.x === minX);

        let glyphCodeDrawn = 0
        let glyphDirectionDrawn = 0
        const methodColorScale = {
            'hue': d3.scaleOrdinal(
                ["ae global", "vac global", "svmw", "sefakmc layerwise", "sefakmc global", "ganspacekmc layerwise", "ganspacekmc global", "vac layerwise", "vac_male layerwise", "vac_female layerwise", "svmw", "va layerwise", "ganspace_male", "ganspace_female"],
                [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 30, 90, 150, 210]
            ),
            'chr': d3.scaleOrdinal(
                ["ae global", "vac global", "svmw", "sefakmc layerwise", "sefakmc global", "ganspacekmc layerwise", "ganspacekmc global", "vac layerwise", "vac_male layerwise", "vac_female layerwise", "svmw", "va layerwise", "ganspace_male", "ganspace_female"],
                [60, 70, 80, 50, 60, 70, 80, 90, 60, 70, 50, 80, 90, 70]
            ),
            'lum': d3.scaleOrdinal(
                ["ae global", "vac global", "svmw", "sefakmc layerwise", "sefakmc global", "ganspacekmc layerwise", "ganspacekmc global", "vac layerwise", "vac_male layerwise", "vac_female layerwise", "svmw", "va layerwise", "ganspace_male", "ganspace_female"],
                [40, 50, 60, 70, 80, 60, 70, 80, 50, 60, 40, 50, 60, 70]
            ),
            'lay': d3.scaleOrdinal(
                ['early', 'middle', 'late'],
                [0, 40, 60]
            ),
        }

        let getIndices = (codeIdx, dirIdx) => {
            let contrib = this.state.contributions.filter(d => d.direction === dirIdx && d.code === codeIdx)
            return contrib
        }

        let getImageLink = (codeIdx, flatIdx, experimentName) => {
            // const experimentName = this.props.experimentNames[treeID]
            const bucketPath = `http://localhost:${this.props.port}/served_data/${experimentName}/`
            return bucketPath + `walked/${codeIdx}-${flatIdx}.jpg`
        }

        let insertImage = (selection, imageLink, imageSize, xPos, yPos) => {
            // Sets up single image
            selection
                .attr('onerror', "this.style.display='none'")
                .attr("xlink:href", imageLink)
                .attr('transform', `translate(${xPos}, ${yPos})`)
                .attr('width', d => imageSize)
                .attr('height', d => imageSize)
                .attr('opacity', 0)
                .on('click', function (event, d) {
                    d3.select(this).raise().transition().duration(500)
                        .attr('width', d3.select(this).attr('width') == imageSize ? imageSize * 4 : imageSize)
                        .attr('height', d3.select(this).attr('height') == imageSize ? imageSize * 4 : imageSize)
                })
                .transition().duration(10) // Adjust the duration as needed
                .attr('opacity', 1) // Final opacity
        }

        let insertCodeGlyphSparkline = (selection, dIdx, width) => {
            let data = [];
            try {
                data = this.state.directionGroupedMagnitude[dIdx][1].map(d => d.mag_contribution);
            } catch (e) {
            }

            // Define scales for x and y
            const xScale = d3.scaleLinear()
                .domain([0, data.length - 1]) // Map indices to width
                .range([0, width]);

            const yScale = d3.scaleLinear()
                // .domain([d3.min(data) || 0, d3.max(data) || 1]) // Map data values to height
                .domain([0, d3.max(data) || 1]) // Map data values to height
                .range([20, 0]); // Inverted for a top-down coordinate system

            // Add a container rect
            selection
                .append('rect')
                .attr('width', width)
                .attr('height', 20)
                .attr('fill', 'none')
                .attr('stroke', 'black');

            // Create area generator
            const areaGenerator = d3.area()
                .x((_, i) => xScale(i)) // X position is based on index
                .y0(20) // Bottom of the area
                .y1(d => yScale(d)) // Top of the area based on data
                .curve(d3.curveBasis); // Smooth curve

            // Create line generator
            const lineGenerator = d3.line()
                .x((_, i) => xScale(i)) // X position is based on index
                .y(d => yScale(d)) // Y position is based on data
                .curve(d3.curveBasis); // Smooth curve

            // Append the filled area
            selection
                .append('path')
                .datum(data) // Bind data
                .attr('d', areaGenerator) // Generate path data for the area
                .attr('fill', 'lightblue') // Fill color for the area
                .attr('opacity', 0.7);

            // Append the sparkline
            selection
                .append('path')
                .datum(data) // Bind data
                .attr('d', lineGenerator) // Generate path data for the line
                .attr('stroke', 'black')
                .attr('stroke-width', 1.5)
                .attr('fill', 'none');
        };

        let insertDirectionGlyphSparkline = (selection, cIdx, width) => {
            let data = [];
            try {
                data = this.state.codeGroupedMagnitude[cIdx][1].map(d => d.mag_contribution);
            } catch (e) {
            }

            // Define scales for x and y
            const yScale = d3.scaleLinear()
                .domain([0, data.length - 1]) // Map indices to height
                .range([0, width]);

            const xScale = d3.scaleLinear()
                // .domain([d3.min(data) || 0, d3.max(data) || 1]) // Map data values to width
                .domain([0, d3.max(data) || 1]) // Map data values to width
                .range([0, 20]);

            // Add a container rect
            selection
                .append('rect')
                .attr('width', 20)
                .attr('height', width)
                .attr('fill', 'none')
                .attr('stroke', 'black');

            // Create area generator
            const areaGenerator = d3.area()
                .x(d => xScale(d)) // X position is based on data
                .y0(width) // Bottom of the area
                .y1((_, i) => yScale(i)) // Top of the area based on index
                .curve(d3.curveBasis); // Smooth curve

            // Create line generator
            const lineGenerator = d3.line()
                .x(d => xScale(d)) // X position is based on data
                .y((_, i) => yScale(i)) // Y position is based on index
                .curve(d3.curveBasis); // Smooth curve

            // Append the filled area
            selection
                .append('path')
                .datum(data) // Bind data
                .attr('d', areaGenerator) // Generate path data for the area
                .attr('fill', 'lightblue') // Fill color for the area
                .attr('opacity', 0.7);

            // Append the sparkline
            selection
                .append('path')
                .datum(data) // Bind data
                .attr('d', lineGenerator) // Generate path data for the line
                .attr('stroke', 'black')
                .attr('stroke-width', 1.5)
                .attr('fill', 'none');
        };

        let insertCodeGlyph = (selection, dIdx, width) => {
            let data = []
            try {
                data = this.state.directionGroupedMagnitude[dIdx][1].map(d => d.mag_contribution)
            } catch (e) {
            }
            console.log("Code glyph data ", data)

            // Define scales for height and width
            const heightScale = d3.scaleLinear()
                // .domain([d3.min(data) || 0, d3.max(data) || 1]) // Avoid NaN issues with empty data
                // .domain([0, d3.max(data) || 1]) // Avoid NaN issues with empty data
                .domain([this.state.magmin, this.state.magmax || 1]) // Avoid NaN issues with empty data
                .range([0, 20]);

            const widthScale = d3.scaleBand()
                .domain(d3.range(data.length)) // Correctly set the domain for band scale
                .range([0, width])
                .padding(0.1); // Add padding for better visuals

            // Add a container rect
            selection
                .append('rect')
                .attr('width', width)
                .attr('height', 20)
                .attr('fill', 'none')
                .attr('stroke', 'black');

            // Append a `g` element to hold the bars
            const barsGroup = selection.append('g');

            // Bind data to bars
            barsGroup.selectAll('rect')
                .data(data)
                .join('rect') // Use join pattern for enter/update/exit handling
                .attr('x', (d, i) => widthScale(i)) // Calculate x position using widthScale
                .attr('y', d => 20 - heightScale(d)) // Align bars to the bottom
                .attr('width', widthScale.bandwidth()) // Set bar width
                .attr('height', d => heightScale(d)) // Set bar height
                .attr('fill', 'black');
        }

        let insertDirectionGlyph = (selection, cIdx, width) => {
            let data = []
            try {
                data = this.state.codeGroupedMagnitude[cIdx][1].map(d => d.mag_contribution)
            } catch (e) {
            }
            // Define scales for height and width
            const widthScale = d3.scaleLinear()
                // .domain([d3.min(data) || 0, d3.max(data) || 1]) // Avoid NaN issues with empty data
                // .domain([0, d3.max(data) || 1]) // Avoid NaN issues with empty data
                .domain([this.state.magmin, this.state.magmax || 1]) // Avoid NaN issues with empty data
                .range([0, 20]);

            const placementScale = d3.scaleBand()
                .domain(d3.range(data.length)) // Correctly set the domain for band scale
                .range([width, 0])
                .padding(0.1); // Add padding for better visuals

            // Add a container rect
            selection
                .append('rect')
                .attr('width', 20)
                .attr('height', width)
                .attr('fill', 'none')
                .attr('stroke', 'black');

            // Append a `g` element to hold the bars
            const barsGroup = selection.append('g');

            // Bind data to bars
            barsGroup.selectAll('rect')
                .data(data)
                .join('rect') // Use join pattern for enter/update/exit handling
                .attr('x', (d, i) => 20 - widthScale(d)) //
                .attr('y', (d, i) => placementScale(i)) // Calculate y position using placementScale
                .attr('width', d => widthScale(d)) // Calculate y position using widthScale
                .attr('height', placementScale.bandwidth()) // Set bar height
                .attr('fill', 'black');
        }

        let okayToDrawDGlyph = (count) => {
            return count < this.state.directionGroupedMagnitude.length;
        }

        let okayToDrawCGlyph = (count) => {
            return count < this.state.codeGroupedMagnitude.length;
        }

        // let topG = selection.filter(d => (d[0].depth === this.state.visDepth && d[1].depth === this.state.visDepth && d[0].leaf)) // Draw only for the top boxes.
        // let topG = selection.filter(d => d[0].depth === this.state.visDepth && d[1].depth === this.state.visDepth) // Draw only for the top boxes.
        // topG.attr('width', d => d[0].size).attr('height', d => d[1].size)

        // Number of topGs
        let topGcount = selection.size()
        let methodDrawnCount = -1

        // scales for individual boxes
        selection.each(function (d, i) {
            let topRow = false
            let leftCol = false
            for (let k in topRowRects) {
                if (d === topRowRects[k].rect) {
                    topRow = true
                }
            }

            for (let k in leftColRects) {
                if (d === leftColRects[k].rect) {
                    leftCol = true
                }
            }


            const width = d[0].size, height = d[1].size
            const selection = d3.select(this)
            let nodeData = selection.data()[0]

            let imageSize = 120

            // Determine how many images can fit into each box.
            let verticalCount = Math.floor(height / imageSize)  // Code budget
            if (topRow) {
                verticalCount = Math.floor((height - 24) / imageSize)  // Code budget
            }

            let horizontalCount = Math.floor(width / imageSize) // Direction budget

            // Use width and height to determine padding.
            const horizontalPadding = (width - (imageSize * horizontalCount)) / 2
            const verticalPadding = (height - (imageSize * verticalCount)) / 2

            // Setup scales
            const horizontalScale = d3.scaleBand().domain(Array.from({length: horizontalCount}, (_, i) => i))
                .range([horizontalPadding, width - horizontalPadding]).paddingOuter(0.1).paddingInner(0.1)

            const verticalScale = d3.scaleBand().domain(Array.from({length: verticalCount}, (_, i) => i))
                .range([verticalPadding, height - verticalPadding]).paddingOuter(0.1).paddingInner(0.1)

            let directionLeaves = accumulateLeafNodesBudget(nodeData[0])
            let codeLeaves = accumulateLeafNodesBudget(nodeData[1])

            // If the budget is larger than available resources, reduce the budget
            if (directionLeaves.length < horizontalCount) {
                horizontalCount = directionLeaves.length
            }
            if (codeLeaves.length < verticalCount) {
                verticalCount = codeLeaves.length
            }

            // Draw Image
            let directionSample = evenlySampleArray(directionLeaves, horizontalCount)
            let codeSample = evenlySampleArray(codeLeaves, verticalCount)
            for (var h = 0; h < horizontalCount; h++) {
                methodDrawnCount += 1
                for (var v = 0; v < verticalCount; v++) {
                    const expName = directionSample[h].expName
                    let imageLink = getImageLink(codeSample[v].name, directionSample[h].flatIdx, expName)
                    if (codeSample[v] && directionSample[h]) {
                        let xPos = horizontalScale(h)
                        let yPos = verticalScale(v)
                        let yPosImg = yPos
                        if (leftCol) xPos += 20
                        if (topRow) yPos += 20

                        selection
                            .append('image')
                            .call(insertImage,
                                imageLink,
                                horizontalScale.bandwidth(),
                                xPos,
                                yPos)

                        // Only when it is the first item in the vertical
                        let imageSizeAdjuster = 15
                        if (topRow && v === 0) {
                            // if (v === 0 && methodDrawnCount < horizontalCount * topGcount) {
                            let [domainName, methodName, applicationName, layerName, layerSubName] = splitExperimentName(expName)
                            methodName = methodName + ' ' + applicationName

                            selection
                                .append('circle')
                                .attr('r', imageSize / imageSizeAdjuster)
                                // .attr('transform', `translate(${xPos + horizontalScale.bandwidth() / 2}, ${yPos - (imageSize / imageSizeAdjuster) - 2})`)
                                // .attr('transform', `translate(${xPos + horizontalScale.bandwidth() - (imageSize / imageSizeAdjuster)}, ${yPos - (imageSize / imageSizeAdjuster) - 2})`)
                                .attr('transform', `translate(${xPos + horizontalScale.bandwidth() / 2}, ${yPos - (imageSize / imageSizeAdjuster) - 24})`)
                                .attr('fill', d => {
                                    let color = d3.hcl(
                                        methodColorScale.hue(methodName),
                                        methodColorScale.chr(methodName),
                                        methodColorScale.lum(methodName) - methodColorScale.lay(layerName))
                                    return color
                                })
                                .style('stroke-width', '3.5') // Add width to the stroke
                                .style('opacity', 0.75)
                                .raise()
                        }
                    }
                }
            }

            if (topRow) {
                for (var h = 0; h < horizontalCount; h++) {
                    let xPos = horizontalScale(h)
                    let yPos = verticalScale(0)
                    if (leftCol) xPos += 20
                    if (topRow) yPos += 20

                    selection
                        .append('g')
                        .attr('transform', `translate(${xPos}, ${yPos - 22})`)
                        // .call(insertCodeGlyph, h, horizontalScale.bandwidth() / 1.2)
                        .call(insertCodeGlyph, h, horizontalScale.bandwidth())
                        // .call(insertCodeGlyphSparkline, h, horizontalScale.bandwidth() / 1.2)
                    glyphDirectionDrawn++
                }

            }

            if (leftCol) {
                for (var v = 0; v < verticalCount; v++) {
                    let xPos = horizontalScale(0)
                    let yPos = verticalScale(v)
                    if (leftCol) xPos += 20
                    if (topRow) yPos += 20

                    selection
                        .append('g')
                        .attr('transform', `translate(${xPos - 22}, ${yPos})`)
                        .call(insertDirectionGlyph, v, horizontalScale.bandwidth())
                        // .call(insertDirectionGlyphSparkline, v, verticalScale.bandwidth())
                    glyphCodeDrawn++
                }

            }
        })

    }

    update_ops = (update) => {
        update
            .attr('transform', d => `translate(${d[0].position},${d[1].position})`)
            .select('rect') // This propagates data attached to group element to a child element, which is 'rect' element.
            .call(this.setup_rects)
    }

    joinBiTree = () => {
        let biColorScale = biTreeColorScale([0, 10], [40, 97], 4)

        // Actual join.
        d3.select(this.gref.current).selectAll('.node')
            .data([[this.state.positionalHierarchyDirection, this.state.positionalHierarchyCode]], d => d[0].name + '-' + d[1].name) // This is a key matching function.
            .join(
                enter => this.enter_ops(enter),
                update => this.update_ops(update),
                exit => exit.remove()
            )

        for (let i = 0; i < this.props.visDepth; i++) {
            let next_depth_g = d3.select(this.gref.current).selectAll('.node').filter(d => {
                return d[0].depth === i
            })
            next_depth_g.selectAll('.node')
                .data(d => {
                    return d3.cross(d[0].children, d[1].children)
                }, d => d[0].name + '-' + d[1].name)
                .join(
                    enter => this.enter_ops(enter),
                    update => this.update_ops(update),
                    exit => exit.remove()
                )
        }
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if ((prevState.positionalHierarchyCode !== this.state.positionalHierarchyCode) ||
            (prevState.positionalHierarchyDirection !== this.state.positionalHierarchyDirection)) {
            this.joinBiTree()
        }
    }

    render() {
        if (!this.state.translate)
            return

        return (
            <g ref={this.gref}
               transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`}
               width={this.state.size[0]}
               height={this.state.size[1]}/>
        )
    }
}