// Service Worker for PWA functionality
const CACHE_NAME = 'portfolio-v2.1.0';
const urlsToCache = [
  '/',
  '/index.html',
  '/styles.css',
  '/styles-advanced.css',
  '/script.js',
  '/i18n.js',
  '/manifest.json',
  '/electronics-projects.html',
  '/photonics-projects.html',
  '/machine-learning-projects.html',
  '/innovation-projects.html',
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

// Fetch event
// HTML / navigations: network-first (so content updates show up immediately, cache is offline fallback).
// Other GETs: cache-first with background refresh.
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') {
    return;
  }

  const isHTML = event.request.mode === 'navigate' ||
    (event.request.destination === 'document') ||
    (event.request.headers.get('accept') || '').includes('text/html');

  if (isHTML) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          if (response && response.status === 200 && response.type !== 'opaque') {
            const copy = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => caches.match(event.request).then(cached => cached || caches.match('/offline.html')))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) {
        return cached;
      }
      return fetch(event.request.clone()).then(response => {
        if (!response || response.status !== 200 || response.type === 'opaque') {
          return response;
        }
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        return response;
      });
    }).catch(error => {
      console.error('Fetch failed:', error);
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
    icon: '/assets/icons/icon-192x192.png',
    badge: '/assets/icons/icon-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Projects'
      },
      {
        action: 'close',
        title: 'Close'
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