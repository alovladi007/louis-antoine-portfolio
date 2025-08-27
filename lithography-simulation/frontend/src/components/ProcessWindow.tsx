'use client'

import { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  Slider,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  Button,
  Tooltip
} from '@mui/material'
import { TrendingUp, TrendingDown, CheckCircle, Warning } from '@mui/icons-material'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
)

interface ProcessMetrics {
  exposureLatitude: number
  depthOfFocus: number
  maskErrorFactor: number
  cdUniformity: number
}

export default function ProcessWindow() {
  const [exposure, setExposure] = useState<number>(1.0)
  const [focus, setFocus] = useState<number>(0)
  const [metrics, setMetrics] = useState<ProcessMetrics>({
    exposureLatitude: 8.5,
    depthOfFocus: 200,
    maskErrorFactor: 2.8,
    cdUniformity: 95.2
  })

  useEffect(() => {
    // Simulate metric updates based on parameters
    const newMetrics = {
      exposureLatitude: 8.5 + (1 - Math.abs(exposure - 1)) * 2 - Math.abs(focus) * 0.01,
      depthOfFocus: 200 - Math.abs(focus) * 2 + (1 - Math.abs(exposure - 1)) * 50,
      maskErrorFactor: 2.8 + Math.abs(exposure - 1) * 0.5 + Math.abs(focus) * 0.005,
      cdUniformity: 95.2 - Math.abs(exposure - 1) * 5 - Math.abs(focus) * 0.05
    }
    setMetrics(newMetrics)
  }, [exposure, focus])

  const chartData = {
    labels: ['0.8', '0.9', '1.0', '1.1', '1.2'],
    datasets: [
      {
        label: 'CD at Best Focus',
        data: [48, 46, 45, 44.5, 44],
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'CD at +50nm Defocus',
        data: [49, 47, 45.5, 45, 45],
        borderColor: '#8B5CF6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'CD at -50nm Defocus',
        data: [47, 45.5, 44.5, 44, 43.5],
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      },
      title: {
        display: false,
      },
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Critical Dimension (nm)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Exposure Dose (relative)'
        }
      }
    }
  }

  const getMetricStatus = (value: number, threshold: number, inverse: boolean = false) => {
    const isGood = inverse ? value < threshold : value > threshold
    return {
      color: isGood ? 'success' : 'warning',
      icon: isGood ? <CheckCircle fontSize="small" /> : <Warning fontSize="small" />,
      trend: isGood ? <TrendingUp fontSize="small" /> : <TrendingDown fontSize="small" />
    }
  }

  return (
    <Box>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Exposure Dose
          </Typography>
          <Slider
            value={exposure}
            onChange={(e, value) => setExposure(value as number)}
            min={0.8}
            max={1.2}
            step={0.01}
            marks={[
              { value: 0.8, label: '0.8' },
              { value: 1.0, label: '1.0' },
              { value: 1.2, label: '1.2' }
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value.toFixed(2)}x`}
          />
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Focus Offset (nm)
          </Typography>
          <Slider
            value={focus}
            onChange={(e, value) => setFocus(value as number)}
            min={-100}
            max={100}
            step={5}
            marks={[
              { value: -100, label: '-100' },
              { value: 0, label: '0' },
              { value: 100, label: '100' }
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}nm`}
          />
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Exposure Latitude', value: metrics.exposureLatitude, unit: '%', threshold: 7 },
          { label: 'Depth of Focus', value: metrics.depthOfFocus, unit: 'nm', threshold: 150 },
          { label: 'MEEF', value: metrics.maskErrorFactor, unit: '', threshold: 3, inverse: true },
          { label: 'CD Uniformity', value: metrics.cdUniformity, unit: '%', threshold: 95 }
        ].map((metric, index) => {
          const status = getMetricStatus(metric.value, metric.threshold, metric.inverse)
          return (
            <Grid item xs={6} key={index}>
              <Paper sx={{ p: 1.5, height: '100%' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    {metric.label}
                  </Typography>
                  <Tooltip title={status.color === 'success' ? 'Within spec' : 'Out of spec'}>
                    <Box>{status.icon}</Box>
                  </Tooltip>
                </Box>
                <Typography variant="h6" fontWeight="bold">
                  {metric.value.toFixed(1)}{metric.unit}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  {status.trend}
                  <Chip
                    label={status.color === 'success' ? 'Good' : 'Warning'}
                    size="small"
                    color={status.color as any}
                    sx={{ height: 20 }}
                  />
                </Box>
              </Paper>
            </Grid>
          )
        })}
      </Grid>

      <Paper sx={{ p: 2, height: 250 }}>
        <Typography variant="subtitle2" gutterBottom>
          CD vs Exposure Dose
        </Typography>
        <Line data={chartData} options={chartOptions} />
      </Paper>

      <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
        <Button variant="outlined" size="small" fullWidth>
          Optimize
        </Button>
        <Button variant="contained" size="small" fullWidth>
          Apply Settings
        </Button>
      </Box>
    </Box>
  )
}