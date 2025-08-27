import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import Navigation from '@/components/Navigation'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Photolithography & Optical Metrology Simulation',
  description: 'Advanced semiconductor lithography simulation and metrology platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <Navigation />
          <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
            {children}
          </main>
          <ToastContainer position="bottom-right" />
        </Providers>
      </body>
    </html>
  )
}