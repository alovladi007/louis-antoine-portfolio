'use client'

import { useState } from 'react'
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Badge,
  Tooltip
} from '@mui/material'
import {
  Menu as MenuIcon,
  Memory,
  GraphicEq,
  BubbleChart,
  Science,
  Dashboard,
  Settings,
  Notifications,
  Person,
  Logout,
  Home,
  Analytics,
  CloudUpload
} from '@mui/icons-material'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

const navigationItems = [
  { title: 'Home', icon: <Home />, path: '/' },
  { title: 'Dashboard', icon: <Dashboard />, path: '/dashboard' },
  { title: 'Lithography', icon: <Memory />, path: '/lithography' },
  { title: 'Metrology', icon: <GraphicEq />, path: '/metrology' },
  { title: 'OPC', icon: <BubbleChart />, path: '/opc' },
  { title: 'Defects', icon: <Science />, path: '/defects' },
  { title: 'Analytics', icon: <Analytics />, path: '/analytics' },
  { title: 'Upload', icon: <CloudUpload />, path: '/upload' },
]

export default function Navigation() {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [notificationAnchor, setNotificationAnchor] = useState<null | HTMLElement>(null)
  const router = useRouter()

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
  }

  const handleNotificationOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchor(event.currentTarget)
  }

  const handleNotificationClose = () => {
    setNotificationAnchor(null)
  }

  const handleLogout = () => {
    // Handle logout logic
    handleMenuClose()
    router.push('/login')
  }

  return (
    <>
      <AppBar position="sticky" elevation={0} sx={{ 
        background: 'linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%)',
        backdropFilter: 'blur(10px)'
      }}>
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={() => setDrawerOpen(true)}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Lithography Simulation Platform
          </Typography>

          <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
            <Button color="inherit" component={Link} href="/lithography">
              Lithography
            </Button>
            <Button color="inherit" component={Link} href="/metrology">
              Metrology
            </Button>
            <Button color="inherit" component={Link} href="/opc">
              OPC
            </Button>
            <Button color="inherit" component={Link} href="/defects">
              Defects
            </Button>
          </Box>

          <Box sx={{ ml: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
            <Tooltip title="Notifications">
              <IconButton color="inherit" onClick={handleNotificationOpen}>
                <Badge badgeContent={3} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Settings">
              <IconButton color="inherit" component={Link} href="/settings">
                <Settings />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Profile">
              <IconButton onClick={handleProfileMenuOpen}>
                <Avatar sx={{ width: 32, height: 32 }}>U</Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <Box sx={{ width: 250 }}>
          <Box sx={{ p: 2, background: 'linear-gradient(45deg, #3B82F6 30%, #8B5CF6 90%)' }}>
            <Typography variant="h6" color="white">
              Navigation
            </Typography>
          </Box>
          <List>
            {navigationItems.map((item) => (
              <ListItem
                button
                key={item.title}
                component={Link}
                href={item.path}
                onClick={() => setDrawerOpen(false)}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.title} />
              </ListItem>
            ))}
            <Divider />
            <ListItem button component={Link} href="/settings">
              <ListItemIcon><Settings /></ListItemIcon>
              <ListItemText primary="Settings" />
            </ListItem>
          </List>
        </Box>
      </Drawer>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
            mt: 1.5,
          },
        }}
      >
        <MenuItem component={Link} href="/profile">
          <ListItemIcon>
            <Person fontSize="small" />
          </ListItemIcon>
          Profile
        </MenuItem>
        <MenuItem component={Link} href="/settings">
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          Settings
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
      </Menu>

      <Menu
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={handleNotificationClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
            mt: 1.5,
            width: 320,
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6">Notifications</Typography>
        </Box>
        <Divider />
        <MenuItem>
          <Box>
            <Typography variant="body2" fontWeight="bold">
              Simulation Complete
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Your lithography simulation has finished processing
            </Typography>
          </Box>
        </MenuItem>
        <MenuItem>
          <Box>
            <Typography variant="body2" fontWeight="bold">
              New Defect Detected
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Critical defect found in wafer batch #2341
            </Typography>
          </Box>
        </MenuItem>
        <MenuItem>
          <Box>
            <Typography variant="body2" fontWeight="bold">
              Process Window Alert
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Process window reduced by 15% in latest run
            </Typography>
          </Box>
        </MenuItem>
      </Menu>
    </>
  )
}