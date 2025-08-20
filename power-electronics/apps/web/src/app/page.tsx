'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Zap, Battery, Sun, Cpu, Activity, 
  Settings, ChevronRight, AlertTriangle 
} from 'lucide-react'

export default function HomePage() {
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null)

  const domains = [
    {
      id: 'ev',
      title: 'EV Traction',
      icon: Battery,
      description: '400V/800V inverters, motor control, DC-DC converters',
      color: 'from-blue-500 to-cyan-500',
      projects: [
        { name: 'Traction Inverter', topology: 'three_phase_inverter' },
        { name: 'OBC Bidirectional', topology: 'bidirectional_dcdc' },
        { name: 'DC-DC Converter', topology: 'interleaved_boost' }
      ]
    },
    {
      id: 'solar',
      title: 'PV/Storage',
      icon: Sun,
      description: 'String inverters, MPPT controllers, battery management',
      color: 'from-yellow-500 to-orange-500',
      projects: [
        { name: 'String Inverter', topology: 'three_phase_inverter' },
        { name: 'MPPT Boost', topology: 'interleaved_boost' },
        { name: 'Battery Charger', topology: 'llc_resonant' }
      ]
    },
    {
      id: 'grid',
      title: 'Grid-Tie',
      icon: Zap,
      description: 'Grid-connected inverters, active filters, V2G systems',
      color: 'from-purple-500 to-pink-500',
      projects: [
        { name: 'Grid Inverter', topology: 'three_phase_inverter' },
        { name: 'Active Filter', topology: 'active_filter' },
        { name: 'V2G System', topology: 'bidirectional_dcdc' }
      ]
    },
    {
      id: 'charger',
      title: 'DC Fast Charging',
      icon: Cpu,
      description: 'CHAdeMO, CCS, high-power charging infrastructure',
      color: 'from-green-500 to-teal-500',
      projects: [
        { name: 'DC Fast Charger', topology: 'llc_resonant' },
        { name: 'Power Module', topology: 'interleaved_boost' },
        { name: 'Isolation Stage', topology: 'llc_resonant' }
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Safety Warning Banner */}
      <div className="bg-yellow-500 text-black px-4 py-2 text-center font-semibold">
        <AlertTriangle className="inline-block w-5 h-5 mr-2" />
        HIGH VOLTAGE WARNING: Equipment operates at dangerous voltage levels. Qualified personnel only.
      </div>

      {/* Header */}
      <header className="border-b border-gray-700 bg-gray-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <nav className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Zap className="w-8 h-8 text-yellow-500" />
              <h1 className="text-2xl font-bold text-white">Power Electronics Platform</h1>
            </div>
            <div className="flex items-center space-x-6">
              <Link href="/projects" className="text-gray-300 hover:text-white transition">
                Projects
              </Link>
              <Link href="/design" className="text-gray-300 hover:text-white transition">
                Design Studio
              </Link>
              <Link href="/simulation" className="text-gray-300 hover:text-white transition">
                Simulation
              </Link>
              <Link href="/bench" className="text-gray-300 hover:text-white transition">
                HIL Bench
              </Link>
              <Link href="/analytics" className="text-gray-300 hover:text-white transition">
                Analytics
              </Link>
              <Link href="/admin" className="text-gray-300 hover:text-white transition">
                <Settings className="w-5 h-5" />
              </Link>
            </div>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center">
          <h2 className="text-5xl font-bold text-white mb-4">
            Advanced Power Electronics
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Design, simulate, and test power converters for EV and renewable energy applications
          </p>
          <div className="flex justify-center space-x-4">
            <Link 
              href="/projects/new"
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-blue-700 transition"
            >
              Start New Project
            </Link>
            <Link 
              href="/docs"
              className="px-6 py-3 bg-gray-700 text-white rounded-lg font-semibold hover:bg-gray-600 transition"
            >
              Documentation
            </Link>
          </div>
        </div>
      </section>

      {/* Domain Cards */}
      <section className="py-12 px-4">
        <div className="container mx-auto">
          <h3 className="text-3xl font-bold text-white mb-8 text-center">Application Domains</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {domains.map((domain) => {
              const Icon = domain.icon
              return (
                <div
                  key={domain.id}
                  className="group cursor-pointer"
                  onClick={() => setSelectedDomain(domain.id === selectedDomain ? null : domain.id)}
                >
                  <div className={`p-6 rounded-xl bg-gradient-to-br ${domain.color} bg-opacity-10 backdrop-blur-sm border border-gray-700 hover:border-gray-500 transition-all`}>
                    <Icon className="w-12 h-12 text-white mb-4" />
                    <h4 className="text-xl font-semibold text-white mb-2">{domain.title}</h4>
                    <p className="text-gray-300 text-sm mb-4">{domain.description}</p>
                    <div className="flex items-center text-white">
                      <span className="text-sm">View Projects</span>
                      <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition" />
                    </div>
                  </div>
                  
                  {selectedDomain === domain.id && (
                    <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                      <h5 className="text-white font-semibold mb-3">Example Projects:</h5>
                      <ul className="space-y-2">
                        {domain.projects.map((project) => (
                          <li key={project.name}>
                            <Link
                              href={`/projects/new?topology=${project.topology}`}
                              className="text-gray-300 hover:text-white flex items-center"
                            >
                              <ChevronRight className="w-4 h-4 mr-2" />
                              {project.name}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-12 px-4 bg-gray-900/50">
        <div className="container mx-auto">
          <h3 className="text-3xl font-bold text-white mb-8 text-center">Platform Capabilities</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 bg-gray-800 rounded-lg">
              <Activity className="w-10 h-10 text-blue-500 mb-4" />
              <h4 className="text-xl font-semibold text-white mb-2">Real-Time Simulation</h4>
              <p className="text-gray-300">
                High-fidelity time-domain simulation with adaptive timestep ODE solvers
              </p>
            </div>
            <div className="p-6 bg-gray-800 rounded-lg">
              <Cpu className="w-10 h-10 text-green-500 mb-4" />
              <h4 className="text-xl font-semibold text-white mb-2">Hardware-in-Loop</h4>
              <p className="text-gray-300">
                CAN-based HIL testing with STM32 control and real power stages
              </p>
            </div>
            <div className="p-6 bg-gray-800 rounded-lg">
              <Settings className="w-10 h-10 text-purple-500 mb-4" />
              <h4 className="text-xl font-semibold text-white mb-2">ML Optimization</h4>
              <p className="text-gray-300">
                Predictive maintenance and control optimization using deep learning
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Stats */}
      <section className="py-12 px-4">
        <div className="container mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-3xl font-bold text-yellow-500">10+</div>
              <div className="text-gray-400">Topologies</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-500">1MHz</div>
              <div className="text-gray-400">Sample Rate</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-500">98%</div>
              <div className="text-gray-400">Peak Efficiency</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-500">10kHz</div>
              <div className="text-gray-400">Control Loop</div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-700 py-8 px-4">
        <div className="container mx-auto text-center text-gray-400">
          <p>© 2024 Power Electronics Platform. MIT License.</p>
          <p className="mt-2 text-sm">
            ⚠️ Safety First: Always follow proper lockout/tagout procedures
          </p>
        </div>
      </footer>
    </div>
  )
}