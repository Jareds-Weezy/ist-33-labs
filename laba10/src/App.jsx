import { useState, useEffect } from 'react'
import QRCode from 'qrcode'
import './App.css'

function App() {
  // Значение по умолчанию согласно варианту задания
  const [text, setText] = useState('Хуснутдинов Руслан Зуфарович')
  const [qrUrl, setQrUrl] = useState('')
  const [size, setSize] = useState(300) // Размер QR кода в пикселях

  // Генерация QR кода при изменении текста или размера
  useEffect(() => {
    generateQR()
  }, [text, size])

  const generateQR = async () => {
    try {
      if (!text.trim()) {
        setQrUrl('') // Очищаем если пусто
        return
      }

      const url = await QRCode.toDataURL(text, {
        width: size,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#ffffff'
        }
      })
      setQrUrl(url)
    } catch (err) {
      console.error(err)
    }
  }

  const handleDownload = () => {
    if (!qrUrl) return
    const link = document.createElement('a')
    link.href = qrUrl
    link.download = 'qrcode.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="app-container">
      <div className="split-layout">
        <div className="left-panel">
          <div className="input-header">
            <span className="label">Обычный текст</span>
            <h1>Создание собственного QR</h1>
          </div>
          <textarea
            className="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Введите текст для генерации QR кода..."
          />
        </div>

        <div className="right-panel">
          <div className="qr-display-area">
            {qrUrl ? (
              <img src={qrUrl} alt="Generated QR Code" className="qr-image" />
            ) : (
              <div className="qr-placeholder">
                Введите текст слева
              </div>
            )}
          </div>

          <div className="controls">
            <div className="control-group">
              <div className="control-label">
                <span>Ширина *</span>
                <span>{size} пикс.</span>
              </div>
              <input
                type="range"
                min="100"
                max="600"
                value={size}
                onChange={(e) => setSize(Number(e.target.value))}
                className="size-slider"
              />
            </div>

            <button onClick={handleDownload} className="download-btn" disabled={!qrUrl}>
              Сохранить PNG
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
