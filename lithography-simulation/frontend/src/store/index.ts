import { configureStore } from '@reduxjs/toolkit'
import simulationReducer from './simulationSlice'
import authReducer from './authSlice'
import uiReducer from './uiSlice'

export const store = configureStore({
  reducer: {
    simulation: simulationReducer,
    auth: authReducer,
    ui: uiReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch