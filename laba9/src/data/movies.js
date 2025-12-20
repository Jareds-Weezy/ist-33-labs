import krakenPoster from '../assets/kraken_v2.png';
import ochiPoster from '../assets/ochi.png';
import zootopiaPoster from '../assets/zootopia.png';

export const movies = [
    {
        id: 1,
        slug: 'kraken',
        title: 'Кракен',
        poster: krakenPoster,
        info: {
            genre: 'Фантастика, Экшн',
            director: 'Николай Лебедев',
            cast: 'Александр Петров, Виктор Добронравов, Сергей Гармаш',
            country: 'Россия',
            premiere: '2025'
        },
        description: 'Атомная подводная лодка российского флота исчезает при загадочных обстоятельствах в водах Баренцева моря. На поиски отправляется опытный командир Виктор Воронин. Ему предстоит столкнуться не только с ледяной бездной, но и с древним ужасом, пробудившимся из глубин, — Кракеном, легендарным морским чудовищем, существование которого считалось мифом.',
        videoThumbnail: krakenPoster,
        videoUrl: 'https://rutube.ru/video/374efd63b4e83a88f1511057ebedb17e/?r=plwd'
    },
    {
        id: 2,
        slug: 'ochi',
        title: 'Очи',
        poster: ochiPoster,
        info: {
            genre: 'Фэнтези, Приключения',
            director: 'Айзея Саксон',
            cast: 'Уиллем Дефо, Эмили Уотсон, Финн Вулфхард',
            country: 'США',
            premiere: '2025'
        },
        description: 'Удивительная история о девочке, которая находит в лесу необычное существо. Очи — таинственные обитатели леса, которых боятся люди. Но героиня решается подружиться с ними, открывая для себя мир магии и гармонии с природой. Это путешествие навсегда изменит её жизнь и жизнь её деревни.',
        videoThumbnail: ochiPoster,
        videoUrl: 'https://s1ru1.kinoplan24.ru/683/0406060506cf96bc8c056bc0/trejler_12.mp4'
    },
    {
        id: 3,
        slug: 'zootopia',
        title: 'Зверополис',
        poster: zootopiaPoster,
        info: {
            genre: 'Мультфильм, Комедия, Детектив',
            director: 'Байрон Ховард, Рич Мур',
            cast: 'Джиннифер Гудвин, Джейсон Бейтман',
            country: 'США',
            premiere: '2016'
        },
        description: 'Зверополис — огромный город, населенный животными всех видов, от слонов до мышей. Джуди Хоппс, крольчиха, поступает на службу в полицию и сразу берется за сложное дело. Ей приходится работать в паре с хитрым лисом Ником Уайлдом, чтобы раскрыть заговор, угрожающий всему городу.',
        videoThumbnail: zootopiaPoster,
        videoUrl: 'https://rutube.ru/video/5115e6107a205e9a65ca2aef6daba4cb/?r=plwd'
    }
];
