// Service Worker for PWA functionality
const CACHE_NAME = 'portfolio-v1.0.0';
const urlsToCache = [
  '/',
  '/index.html',
  '/styles.css',
  '/additional-projects.html',
  '/innovation-projects.html',
  '/portfolio-enhancements.html',
  '/manifest.json',
  '/theme-toggle.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
];

// Install event - cache resources
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
      .catch(err => {
        console.error('Cache installation failed:', err);
      })
  );
  // Force the waiting service worker to become the active service worker
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  // Claim all clients
  self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // Clone the request
        const fetchRequest = event.request.clone();

        return fetch(fetchRequest).then(response => {
          // Check if valid response
          if (!response || response.status !== 200 || response.type === 'opaque') {
            return response;
          }

          // Clone the response
          const responseToCache = response.clone();

          // Cache the fetched response for future use
          caches.open(CACHE_NAME)
            .then(cache => {
              // Don't cache POST requests or non-GET requests
              if (event.request.method === 'GET') {
                cache.put(event.request, responseToCache);
              }
            });

          return response;
        });
      })
      .catch(error => {
        // Offline fallback
        console.error('Fetch failed:', error);
        
        // Return offline page for navigation requests
        if (event.request.mode === 'navigate') {
          return caches.match('/offline.html');
        }
      })
  );
});

// Background sync for offline form submissions
self.addEventListener('sync', event => {
  if (event.tag === 'sync-messages') {
    event.waitUntil(syncMessages());
  }
});

async function syncMessages() {
  try {
    const messages = await getMessagesFromIndexedDB();
    for (const message of messages) {
      await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(message)
      });
      await removeMessageFromIndexedDB(message.id);
    }
  } catch (error) {
    console.error('Sync failed:', error);
  }
}

// Push notifications
self.addEventListener('push', event => {
  const options = {
    body: event.data ? event.data.text() : 'New update available!',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Projects',
        icon: '/icons/checkmark.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icons/xmark.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Portfolio Update', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', event => {
  event.notification.close();

  if (event.action === 'explore') {
    // Open projects page
    event.waitUntil(
      clients.openWindow('/#projects')
    );
  } else {
    // Open home page
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Helper functions for IndexedDB (for offline form storage)
function getMessagesFromIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('PortfolioDB', 1);
    
    request.onsuccess = event => {
      const db = event.target.result;
      const transaction = db.transaction(['messages'], 'readonly');
      const objectStore = transaction.objectStore('messages');
      const getAllRequest = objectStore.getAll();
      
      getAllRequest.onsuccess = () => {
        resolve(getAllRequest.result);
      };
    };
    
    request.onerror = () => {
      reject('Failed to open IndexedDB');
    };
  });
}

function removeMessageFromIndexedDB(id) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('PortfolioDB', 1);
    
    request.onsuccess = event => {
      const db = event.target.result;
      const transaction = db.transaction(['messages'], 'readwrite');
      const objectStore = transaction.objectStore('messages');
      const deleteRequest = objectStore.delete(id);
      
      deleteRequest.onsuccess = () => {
        resolve();
      };
    };
    
    request.onerror = () => {
      reject('Failed to delete from IndexedDB');
    };
  });
}