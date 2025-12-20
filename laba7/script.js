// Переключение выпадающего меню
const toggleBtn = document.getElementById('toggleView');
const dropdownMenu = document.getElementById('dropdownMenu');

toggleBtn.addEventListener('click', function() {
    dropdownMenu.classList.toggle('active');
});

// Закрытие меню при клике вне его
document.addEventListener('click', function(event) {
    if (!event.target.closest('.view-toggle')) {
        dropdownMenu.classList.remove('active');
    }
});

// Ручное переключение представлений
const viewButtons = document.querySelectorAll('.dropdown-menu button');

viewButtons.forEach(button => {
    button.addEventListener('click', function() {
        const view = this.getAttribute('data-view');
        
        if (view === 'mobile') {
            document.body.classList.add('mobile-view');
            toggleBtn.innerHTML = '<span>Mobile</span> <span>▼</span>';
        } else {
            document.body.classList.remove('mobile-view');
            toggleBtn.innerHTML = '<span>Desktop</span> <span>▼</span>';
        }
        
        dropdownMenu.classList.remove('active');
    });
});

// Автоматическое переключение при изменении размера окна
function handleResize() {
    if (window.innerWidth <= 768) {
        document.body.classList.add('mobile-view');
        toggleBtn.innerHTML = '<span>Mobile</span> <span>▼</span>';
    } else {
        document.body.classList.remove('mobile-view');
        toggleBtn.innerHTML = '<span>Desktop</span> <span>▼</span>';
    }
}

// Инициализация при загрузке
window.addEventListener('load', handleResize);
window.addEventListener('resize', handleResize);