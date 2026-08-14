import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import PageContainer from './components/PageContainer'
import Home from './pages/Home'
import Upload from './pages/Upload'
import Analysis from './pages/Analysis'

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-[#0F172A] text-[#F8FAFC]">
        <Navbar />
        <main className="flex-grow">
          <PageContainer>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/analysis" element={<Analysis />} />
            </Routes>
          </PageContainer>
        </main>
      </div>
    </Router>
  )
}

export default App
