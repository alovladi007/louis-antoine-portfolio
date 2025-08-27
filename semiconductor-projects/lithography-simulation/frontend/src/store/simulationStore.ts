import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import axios from 'axios'

interface SimulationData {
  id: string
  type: string
  status: 'idle' | 'running' | 'completed' | 'failed'
  progress: number
  results: any
  parameters: any
  createdAt: Date
  completedAt?: Date
}

interface SimulationStore {
  simulations: SimulationData[]
  currentSimulation: SimulationData | null
  isRunning: boolean
  simulationData: any
  
  // Actions
  startSimulation: (type: string, parameters: any) => Promise<void>
  stopSimulation: (id: string) => void
  updateProgress: (id: string, progress: number) => void
  setCurrentSimulation: (simulation: SimulationData | null) => void
  fetchSimulations: () => Promise<void>
  deleteSimulation: (id: string) => Promise<void>
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const useSimulationStore = create<SimulationStore>()(
  devtools(
    persist(
      (set, get) => ({
        simulations: [],
        currentSimulation: null,
        isRunning: false,
        simulationData: null,

        startSimulation: async (type, parameters) => {
          try {
            set({ isRunning: true })
            
            const response = await axios.post(`${API_URL}/api/${type}/simulate`, parameters, {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              }
            })

            const newSimulation: SimulationData = {
              id: response.data.simulation_id,
              type,
              status: 'running',
              progress: 0,
              results: null,
              parameters,
              createdAt: new Date()
            }

            set(state => ({
              simulations: [...state.simulations, newSimulation],
              currentSimulation: newSimulation
            }))

            // Start WebSocket connection for progress updates
            const ws = new WebSocket(`${process.env.NEXT_PUBLIC_WS_URL}/ws/simulation/${response.data.simulation_id}`)
            
            ws.onmessage = (event) => {
              const data = JSON.parse(event.data)
              get().updateProgress(response.data.simulation_id, data.progress)
              
              if (data.status === 'completed') {
                set(state => ({
                  isRunning: false,
                  simulations: state.simulations.map(sim =>
                    sim.id === response.data.simulation_id
                      ? { ...sim, status: 'completed', progress: 100, completedAt: new Date() }
                      : sim
                  )
                }))
                ws.close()
              }
            }

            ws.onerror = () => {
              set({ isRunning: false })
              ws.close()
            }

          } catch (error) {
            console.error('Failed to start simulation:', error)
            set({ isRunning: false })
            throw error
          }
        },

        stopSimulation: (id) => {
          set(state => ({
            isRunning: false,
            simulations: state.simulations.map(sim =>
              sim.id === id ? { ...sim, status: 'idle' } : sim
            )
          }))
        },

        updateProgress: (id, progress) => {
          set(state => ({
            simulations: state.simulations.map(sim =>
              sim.id === id ? { ...sim, progress } : sim
            ),
            currentSimulation: state.currentSimulation?.id === id
              ? { ...state.currentSimulation, progress }
              : state.currentSimulation
          }))
        },

        setCurrentSimulation: (simulation) => {
          set({ currentSimulation: simulation })
        },

        fetchSimulations: async () => {
          try {
            const response = await axios.get(`${API_URL}/api/simulations`, {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              }
            })
            
            set({ simulations: response.data })
          } catch (error) {
            console.error('Failed to fetch simulations:', error)
          }
        },

        deleteSimulation: async (id) => {
          try {
            await axios.delete(`${API_URL}/api/simulations/${id}`, {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              }
            })
            
            set(state => ({
              simulations: state.simulations.filter(sim => sim.id !== id),
              currentSimulation: state.currentSimulation?.id === id ? null : state.currentSimulation
            }))
          } catch (error) {
            console.error('Failed to delete simulation:', error)
            throw error
          }
        }
      }),
      {
        name: 'simulation-storage',
        partialize: (state) => ({ 
          simulations: state.simulations,
          currentSimulation: state.currentSimulation
        })
      }
    )
  )
)