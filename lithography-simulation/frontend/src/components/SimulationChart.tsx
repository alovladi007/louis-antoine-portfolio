'use client'

import { useEffect, useRef, useState } from 'react'
import Plot from 'react-plotly.js'
import { Box, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { useSimulationStore } from '@/store/simulationStore'

type ChartType = 'aerial' | 'resist' | 'process' | '3d'

export default function SimulationChart() {
  const [chartType, setChartType] = useState<ChartType>('aerial')
  const { simulationData, isRunning } = useSimulationStore()
  
  const generateAerialImageData = () => {
    // Generate sample aerial image data
    const size = 100
    const z = []
    for (let i = 0; i < size; i++) {
      const row = []
      for (let j = 0; j < size; j++) {
        const value = Math.sin(i / 10) * Math.cos(j / 10) + 
                     Math.random() * 0.1 +
                     (Math.abs(i - 50) < 10 && Math.abs(j - 50) < 20 ? 1 : 0)
        row.push(value)
      }
      z.push(row)
    }
    return z
  }

  const generateResistProfile = () => {
    const x = []
    const y = []
    for (let i = 0; i < 100; i++) {
      x.push(i)
      y.push(1 / (1 + Math.exp(-0.2 * (i - 50))) + Math.random() * 0.05)
    }
    return { x, y }
  }

  const generateProcessWindow = () => {
    const exposure = []
    const focus = []
    const cd = []
    
    for (let e = 0.8; e <= 1.2; e += 0.05) {
      for (let f = -100; f <= 100; f += 20) {
        exposure.push(e)
        focus.push(f)
        cd.push(45 + (e - 1) * 10 - Math.abs(f) * 0.05 + Math.random() * 2)
      }
    }
    
    return { exposure, focus, cd }
  }

  const renderChart = () => {
    switch (chartType) {
      case 'aerial':
        return (
          <Plot
            data={[
              {
                z: generateAerialImageData(),
                type: 'heatmap',
                colorscale: 'Viridis',
                colorbar: {
                  title: 'Intensity',
                  titleside: 'right'
                }
              }
            ]}
            layout={{
              title: 'Aerial Image Simulation',
              xaxis: { title: 'X Position (nm)' },
              yaxis: { title: 'Y Position (nm)' },
              autosize: true,
              margin: { l: 50, r: 50, t: 50, b: 50 }
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '400px' }}
          />
        )
      
      case 'resist':
        const resistData = generateResistProfile()
        return (
          <Plot
            data={[
              {
                x: resistData.x,
                y: resistData.y,
                type: 'scatter',
                mode: 'lines',
                name: 'Resist Profile',
                line: { color: '#3B82F6', width: 3 }
              },
              {
                x: resistData.x,
                y: resistData.x.map(() => 0.5),
                type: 'scatter',
                mode: 'lines',
                name: 'Threshold',
                line: { color: '#EF4444', width: 2, dash: 'dash' }
              }
            ]}
            layout={{
              title: 'Photoresist Development Profile',
              xaxis: { title: 'Position (nm)' },
              yaxis: { title: 'Normalized Intensity' },
              autosize: true,
              margin: { l: 50, r: 50, t: 50, b: 50 },
              showlegend: true
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '400px' }}
          />
        )
      
      case 'process':
        const processData = generateProcessWindow()
        return (
          <Plot
            data={[
              {
                x: processData.exposure,
                y: processData.focus,
                z: processData.cd,
                type: 'contour',
                colorscale: 'RdBu',
                reversescale: true,
                contours: {
                  start: 40,
                  end: 50,
                  size: 1
                },
                colorbar: {
                  title: 'CD (nm)',
                  titleside: 'right'
                }
              }
            ]}
            layout={{
              title: 'Process Window Analysis',
              xaxis: { title: 'Exposure Dose (relative)' },
              yaxis: { title: 'Focus Offset (nm)' },
              autosize: true,
              margin: { l: 60, r: 50, t: 50, b: 50 }
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '400px' }}
          />
        )
      
      case '3d':
        return (
          <Plot
            data={[
              {
                z: generateAerialImageData(),
                type: 'surface',
                colorscale: 'Viridis',
                contours: {
                  z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "#42f462",
                    project: { z: true }
                  }
                }
              }
            ]}
            layout={{
              title: '3D Surface Profile',
              scene: {
                xaxis: { title: 'X (nm)' },
                yaxis: { title: 'Y (nm)' },
                zaxis: { title: 'Height (nm)' },
                camera: {
                  eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
              },
              autosize: true,
              margin: { l: 0, r: 0, t: 30, b: 0 }
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '400px' }}
          />
        )
      
      default:
        return null
    }
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="body2" color="text.secondary">
          {isRunning ? 'Simulation Running...' : 'Ready'}
        </Typography>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(e, value) => value && setChartType(value)}
          size="small"
        >
          <ToggleButton value="aerial">Aerial</ToggleButton>
          <ToggleButton value="resist">Resist</ToggleButton>
          <ToggleButton value="process">Process</ToggleButton>
          <ToggleButton value="3d">3D</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      
      <Box className="chart-container">
        {renderChart()}
      </Box>
    </Box>
  )
}