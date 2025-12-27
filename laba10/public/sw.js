const CACHE_NAME = 'qr-pwa-v2';
const STATIC_ASSETS = [
    './',
    './index.html',
    './manifest.json',
    './icons/icon-192x192.png',
    './icons/icon-512x512.png'
];

self.addEventListener('install', (event) => {
    // Предварительное кэширование статических ассетов
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    // Очистка старых кэшей
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cache) => {
                    if (cache !== CACHE_NAME) {
                        return caches.delete(cache);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            // Если есть в кэше — возвращаем
            if (response) {
                return response;
            }

            // Иначе делаем запрос к сети
            return fetch(event.request).then((response) => {
                // Проверяем валидность ответа
                if (!response || response.status !== 200 || response.type !== 'basic') {
                    return response;
                }

                // Кэшируем новый ресурс
                const responseToCache = response.clone();
                caches.open(CACHE_NAME).then((cache) => {
                    cache.put(event.request, responseToCache);
                });

                return response;
            });
        })
    );
});
