'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Container, 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Button,
  Box,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Paper
} from '@mui/material'
import {
  Science,
  Memory,
  GraphicEq,
  BubbleChart,
  PlayArrow,
  Dashboard,
  Settings,
  TrendingUp,
  Visibility,
  CloudUpload
} from '@mui/icons-material'
import Link from 'next/link'
import SimulationChart from '@/components/SimulationChart'
import ProcessWindow from '@/components/ProcessWindow'
import { useSimulationStore } from '@/store/simulationStore'

const features = [
  {
    title: 'Lithography Simulation',
    description: 'Model mask patterns, aerial images, and resist profiles with nanometer precision',
    icon: <Memory />,
    color: '#3B82F6',
    link: '/lithography'
  },
  {
    title: 'Optical Metrology',
    description: 'Scatterometry, interferometry, and ellipsometry measurements',
    icon: <GraphicEq />,
    color: '#8B5CF6',
    link: '/metrology'
  },
  {
    title: 'OPC Processing',
    description: 'Advanced Optical Proximity Correction with AI-driven optimization',
    icon: <BubbleChart />,
    color: '#10B981',
    link: '/opc'
  },
  {
    title: 'Defect Detection',
    description: 'ML-powered defect classification and yield prediction',
    icon: <Science />,
    color: '#F59E0B',
    link: '/defects'
  }
]

const stats = [
  { label: 'Resolution', value: '7nm', trend: '+15%' },
  { label: 'Process Window', value: '±8%', trend: '+22%' },
  { label: 'Defect Detection', value: '99.2%', trend: '+5%' },
  { label: 'Simulation Speed', value: '10x', trend: '+50%' }
]

export default function HomePage() {
  const [activeSimulation, setActiveSimulation] = useState<string | null>(null)
  const { currentSimulation, isRunning } = useSimulationStore()

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h2" component="h1" gutterBottom fontWeight="bold">
            Photolithography & Optical Metrology
          </Typography>
          <Typography variant="h5" color="text.secondary" sx={{ mb: 4 }}>
            Advanced Semiconductor Manufacturing Simulation Platform
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 4 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              component={Link}
              href="/lithography"
              sx={{ 
                background: 'linear-gradient(45deg, #3B82F6 30%, #8B5CF6 90%)',
                color: 'white'
              }}
            >
              Start Simulation
            </Button>
            <Button
              variant="outlined"
              size="large"
              startIcon={<Dashboard />}
              component={Link}
              href="/dashboard"
            >
              View Dashboard
            </Button>
          </Box>
        </Box>

        {/* Stats Section */}
        <Grid container spacing={3} sx={{ mb: 6 }}>
          {stats.map((stat, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h3" fontWeight="bold" color="primary">
                    {stat.value}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {stat.label}
                  </Typography>
                  <Chip 
                    label={stat.trend} 
                    color="success" 
                    size="small" 
                    sx={{ mt: 1 }}
                  />
                </Paper>
              </motion.div>
            </Grid>
          ))}
        </Grid>

        {/* Features Grid */}
        <Grid container spacing={4} sx={{ mb: 6 }}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={6} lg={3} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 + 0.3 }}
                whileHover={{ scale: 1.05 }}
              >
                <Card 
                  sx={{ 
                    height: '100%',
                    cursor: 'pointer',
                    transition: 'all 0.3s',
                    '&:hover': {
                      boxShadow: 6,
                      transform: 'translateY(-4px)'
                    }
                  }}
                  component={Link}
                  href={feature.link}
                >
                  <CardContent>
                    <Box 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        mb: 2,
                        color: feature.color 
                      }}
                    >
                      {feature.icon}
                      <Typography variant="h6" sx={{ ml: 1 }}>
                        {feature.title}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>

        {/* Live Simulation Preview */}
        <Grid container spacing={4}>
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h5">Live Simulation Preview</Typography>
                  <Box>
                    <Tooltip title="Settings">
                      <IconButton>
                        <Settings />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Full Screen">
                      <IconButton>
                        <Visibility />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                
                {isRunning && <LinearProgress sx={{ mb: 2 }} />}
                
                <SimulationChart />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Process Window Analysis
                </Typography>
                <ProcessWindow />
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Quick Actions */}
        <Box sx={{ mt: 6, textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Quick Actions
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="outlined"
              startIcon={<CloudUpload />}
              component={Link}
              href="/upload"
            >
              Upload Mask Pattern
            </Button>
            <Button
              variant="outlined"
              startIcon={<TrendingUp />}
              component={Link}
              href="/analytics"
            >
              View Analytics
            </Button>
            <Button
              variant="outlined"
              startIcon={<Settings />}
              component={Link}
              href="/settings"
            >
              Configure System
            </Button>
          </Box>
        </Box>
      </motion.div>
    </Container>
  )
}