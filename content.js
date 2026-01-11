// Функция захвата признаков товаров
function extractFeatures() {
    const cards = document.querySelectorAll('.product-card__wrapper');
    return Array.from(cards).map((card, index) => {
        // Селекторы актуализированы под структуру WB 2026
        const rating = parseFloat(card.querySelector('.address-rate-mini')?.innerText || 0);
        const reviews = parseInt(card.querySelector('.product-card__count')?.innerText.replace(/\D/g, '') || 0);
        const price = parseInt(card.querySelector('.price__lower-price')?.innerText.replace(/\D/g, '') || 0);
        
        return {
            element: card,
            features: [index + 1, rating, reviews, price]
        };
    });
}

async function analyze() {
    // 0. Инициализация бэкенда (критично для раздельных библиотек core и cpu)
    if (tf.getBackend() !== 'cpu') {
        await tf.setBackend('cpu');
    }
    await tf.ready();

    const items = extractFeatures();
    if (items.length < 5) return;

    // Использование tf.tidy предотвращает утечки памяти в браузере
    tf.tidy(() => {
        // 1. Превращаем данные в тензор [N, 4]
        const rawTensor = tf.tensor2d(items.map(i => i.features));

        // 2. Нормализация в функциональном стиле (для tf-core)
        const min = tf.min(rawTensor, 0);
        const max = tf.max(rawTensor, 0);
        
        // normalized = (raw - min) / (max - min + epsilon)
        const denum = tf.add(tf.sub(max, min), tf.scalar(1e-6));
        const normalized = tf.div(tf.sub(rawTensor, min), denum);

        // 3. Вычисление веса аномалии
        const weights = tf.tensor1d([-1.0, 0.2, 0.8, -0.1]); 
        
        // Скалярное произведение через tf.matMul
        const scores = tf.reshape(
            tf.matMul(normalized, tf.reshape(weights, [-1, 1])), 
            [-1]
        );
        
        // Статистика через функциональные вызовы
        const mean = tf.mean(scores);
        const std = tf.sqrt(tf.mean(tf.square(tf.sub(scores, mean))));
        
        // Z-score
        const zScoresTensor = tf.div(tf.sub(scores, mean), std);
        const zScores = zScoresTensor.dataSync();

        // 4. Помечаем аномалии
        items.forEach((item, i) => {
            // Если карточка уже помечена, пропускаем
            if (item.element.getAttribute('data-tensor-analyzed')) return;

            // Порог -1.5: позиция значительно выше (лучше), чем должна быть по статистике отзывов/рейтинга
            if (zScores[i] < -1) { 
                item.element.style.outline = "4px dashed #ff4757";
                item.element.style.position = "relative";
                item.element.setAttribute('data-tensor-analyzed', 'true');
                
                const label = document.createElement('span');
                label.innerText = "PROMOTED (AI DETECTED)";
                label.style = "position:absolute;top:0;left:0;background:#ff4757;color:white;font-weight:bold;z-index:100;padding:4px;font-size:10px;pointer-events:none;";
                item.element.appendChild(label);
            }
        });
    });
}

// Запуск с защитой от слишком частых вызовов (Debounce)
// let timeout;
// const observer = new MutationObserver(() => {
//     clearTimeout(timeout);
//     timeout = setTimeout(analyze, 1000);
// });

// // Наблюдаем за изменением списка товаров (для Infinite Scroll)
// const catalog = document.querySelector('.catalog-page__content, .product-card-list');
// if (catalog) {
//     observer.observe(catalog, { childList: true, subtree: true });
// }
// Запуск при загрузке и динамической подгрузке (Infinite Scroll)
let timeout;
window.addEventListener('scroll', () => {
    clearTimeout(timeout);
    timeout = setTimeout(analyze, 1000);
});
analyze();
