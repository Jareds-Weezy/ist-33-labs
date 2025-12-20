import React, { useState, useEffect } from 'react';
import MovieCard from './components/MovieCard';
import { movies } from './data/movies';

function App() {
  const [currentMovie, setCurrentMovie] = useState(null);

  useEffect(() => {
    // Basic randomization: pick one of the 3 movies
    const randomIndex = Math.floor(Math.random() * movies.length);
    setCurrentMovie(movies[randomIndex]);
  }, []);

  return (
    <div className="app-container">
      {currentMovie && <MovieCard movie={currentMovie} />}
    </div>
  );
}

export default App;
