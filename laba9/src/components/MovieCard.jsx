import React, { useState } from 'react';
import './MovieCard.css';

const MovieCard = ({ movie }) => {
    const [isPlaying, setIsPlaying] = useState(false);

    if (!movie) return null;

    const handlePlayClick = () => {
        setIsPlaying(true);
    };

    const renderVideoPlayer = () => {
        const { videoUrl } = movie;

        if (videoUrl.endsWith('.mp4')) {
            return (
                <video
                    className="video-frame"
                    src={videoUrl}
                    controls
                    autoPlay
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
            );
        }

        let embedUrl = videoUrl;
        // Handle Rutube
        if (videoUrl.includes('rutube.ru')) {
            // Extract ID from https://rutube.ru/video/ID/ or similar
            const match = videoUrl.match(/video\/([a-zA-Z0-9]+)/);
            if (match && match[1]) {
                embedUrl = `https://rutube.ru/play/embed/${match[1]}`;
            }
        }
        // Handle YouTube (simple check)
        else if (videoUrl.includes('youtube.com') || videoUrl.includes('youtu.be')) {
            if (!videoUrl.includes('embed')) {
                // Very basic conversion, assuming standard format
                const v = videoUrl.split('v=')[1]?.split('&')[0];
                if (v) embedUrl = `https://www.youtube.com/embed/${v}`;
            }
            // Add autoplay
            embedUrl += (embedUrl.includes('?') ? '&' : '?') + 'autoplay=1';
        }

        return (
            <iframe
                className="video-frame"
                src={embedUrl}
                title="Video player"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
                allowFullScreen
            ></iframe>
        );
    };

    return (
        <div className="movie-card">
            <div className="top-section">
                <div className="poster">
                    <img src={movie.poster} alt={`${movie.title} Poster`} />
                </div>

                <div className="info-column">
                    <h1 className="movie-title">{movie.title}</h1>

                    <div className="info-grid">
                        <span className="info-label">Жанр:</span>
                        <span className="info-value">{movie.info.genre}</span>

                        <span className="info-label">Режиссер:</span>
                        <span className="info-value">{movie.info.director}</span>

                        <span className="info-label">В ролях:</span>
                        <span className="info-value">{movie.info.cast}</span>

                        <span className="info-label">Страна:</span>
                        <span className="info-value">{movie.info.country}</span>

                        <span className="info-label">Премьера:</span>
                        <span className="info-value">{movie.info.premiere}</span>
                    </div>

                    <button className="buy-button">
                        КУПИТЬ БИЛЕТ
                    </button>
                </div>
            </div>

            <div className="description-section">
                <h2>Описание фильма</h2>
                <p className="description-text">{movie.description}</p>
            </div>

            <div className="trailer-section">
                <div className="video-wrapper">
                    {!isPlaying ? (
                        <div className="video-thumbnail-container" onClick={handlePlayClick}>
                            <img
                                src={movie.videoThumbnail}
                                alt="Trailer thumbnail"
                                className="video-placeholder"
                            />
                            <div className="play-button-overlay">
                                <div className="play-icon"></div>
                            </div>
                        </div>
                    ) : (
                        renderVideoPlayer()
                    )}
                </div>
            </div>
        </div>
    );
};

export default MovieCard;
