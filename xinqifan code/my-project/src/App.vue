<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

// 后端地址，可通过 .env 中的 VITE_API_BASE_URL 覆盖
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080'

interface HistoryItem {
  id: number
  created_at: string
  image_filename: string
  image_path: string
  question: string
  answer: string
}

const fileInputRef = ref<HTMLInputElement | null>(null)
const selectedFile = ref<File | null>(null)
const previewUrl = ref('')
const question = ref('')
const answer = ref('')
const loading = ref(false)
const error = ref('')
const history = ref<HistoryItem[]>([])

function triggerFileInput() {
  fileInputRef.value?.click()
}

function setFile(file: File | undefined) {
  if (!file) return
  if (!file.type.startsWith('image/')) {
    error.value = '请上传图片文件'
    return
  }
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  error.value = ''
}

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  setFile(target.files?.[0])
}

function onDrop(e: DragEvent) {
  setFile(e.dataTransfer?.files?.[0])
}

async function submit() {
  if (!selectedFile.value || !question.value) return
  loading.value = true
  error.value = ''
  answer.value = ''
  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)
    formData.append('question', question.value)
    const { data } = await axios.post(`${API_BASE}/api/vqa`, formData)
    answer.value = data.answer
    await fetchHistory()
  } catch (err) {
    if (axios.isAxiosError(err)) {
      error.value = err.response?.data?.detail || '请求失败，请检查后端服务是否已启动'
    } else {
      error.value = '未知错误'
    }
  } finally {
    loading.value = false
  }
}

function copyResult() {
  if (!answer.value) return
  navigator.clipboard.writeText(answer.value)
}

async function fetchHistory() {
  try {
    const { data } = await axios.get(`${API_BASE}/api/history`)
    history.value = data.history
  } catch {
    // 历史记录加载失败时静默忽略，不影响主流程
  }
}

async function removeHistory(id: number) {
  try {
    await axios.delete(`${API_BASE}/api/history/${id}`)
    history.value = history.value.filter((h) => h.id !== id)
  } catch {
    // 删除失败时静默忽略
  }
}

onMounted(() => {
  fetchHistory()
})
</script>

<template>
  <div class="page">
    <header class="app-header">
      <div class="app-header-icon"><i class="ti ti-microscope"></i></div>
      <div>
        <h1 class="app-title">肠道内镜影像 VQA 系统</h1>
        <p class="app-subtitle">上传内镜图像并输入问题，AI 模型将基于图像内容给出分析结果</p>
      </div>
    </header>

    <div class="grid">
      <section class="panel">
        <div class="panel-header">
          <p class="panel-title">上传与提问</p>
        </div>
        <div
          class="upload-box"
          :class="{ 'has-image': previewUrl }"
          @click="triggerFileInput"
          @dragover.prevent
          @drop.prevent="onDrop"
        >
          <input
            ref="fileInputRef"
            type="file"
            accept="image/jpeg,image/png"
            class="hidden-input"
            @change="onFileChange"
          />
          <template v-if="previewUrl">
            <img :src="previewUrl" class="preview-img" alt="预览图" />
            <p class="upload-sub">点击重新选择图片</p>
          </template>
          <template v-else>
            <div class="upload-icon"><i class="ti ti-upload"></i></div>
            <p class="upload-title">上传内镜图像</p>
            <p class="upload-sub">点击或拖拽上传图片（支持 JPG/PNG）</p>
          </template>
        </div>

        <p class="question-label">输入您的问题</p>
        <div class="question-input">
          <i class="ti ti-message-circle"></i>
          <input
            v-model="question"
            type="text"
            class="question-text-input"
            placeholder="例如：该区域是否存在息肉？是否有溃疡？病变范围如何？"
          />
        </div>

        <p v-if="error" class="error-text">{{ error }}</p>

        <button class="submit-btn" :disabled="loading || !selectedFile || !question" @click="submit">
          <i class="ti" :class="loading ? 'ti-loader-2' : 'ti-player-play'"></i>
          {{ loading ? '分析中...' : '提交分析' }}
        </button>
      </section>

      <section class="panel">
        <div class="panel-header">
          <p class="panel-title">AI 分析结果</p>
          <span class="model-badge"><i class="ti ti-settings"></i>模型：Qwen2.5-VL-3B (Kvasir-VQA)</span>
        </div>
        <div class="result-card">
          <template v-if="answer">
            <p class="section-label">诊断结果</p>
            <div class="diagnosis-pill"><i class="ti ti-alert-triangle"></i>{{ answer }}</div>
            <div class="result-footer">
              <button class="copy-btn" @click="copyResult"><i class="ti ti-copy"></i>复制结果</button>
            </div>
          </template>
          <p v-else class="section-label placeholder-text">暂无结果，请上传图像并输入问题后提交分析</p>
        </div>

        <div class="history-section" v-if="history.length">
          <p class="panel-title history-title">历史记录</p>
          <div class="history-list">
            <div class="history-item" v-for="item in history" :key="item.id">
              <div class="history-info">
                <p class="history-q">Q: {{ item.question }}</p>
                <p class="history-a">A: {{ item.answer }}</p>
                <p class="history-time">{{ item.created_at }}</p>
              </div>
              <button class="delete-btn" @click="removeHistory(item.id)"><i class="ti ti-trash"></i></button>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  color: #1F1F1D;
  background-color: #F7F7F5;
  min-height: 100vh;
}
.page { max-width: 1080px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }

.app-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 2rem;
}
.app-header-icon {
  flex-shrink: 0;
  width: 48px; height: 48px;
  border-radius: 12px;
  background: #E1F5EE;
  display: flex; align-items: center; justify-content: center;
}
.app-header-icon i { font-size: 22px; color: #0F6E56; }
.app-title { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
.app-subtitle { font-size: 13px; color: #6B6A64; margin: 0; }

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  align-items: start;
}
.panel {
  display: flex;
  flex-direction: column;
  background: #FFFFFF;
  border: 0.5px solid #E4E2D9;
  border-radius: 14px;
  box-shadow: 0 1px 3px rgba(31,31,29,0.04);
  padding: 1.5rem;
}
.panel-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 8px;
}
.panel-title { font-weight: 600; font-size: 15px; margin: 0; }
.upload-box {
  border: 2px dashed #C9C7BE;
  border-radius: 12px;
  padding: 3rem 1.5rem;
  text-align: center;
  background: #FAFAF8;
  cursor: pointer;
  overflow: hidden;
  transition: border-color 0.15s ease, background-color 0.15s ease;
}
.upload-box:hover { border-color: #1D9E75; background: #F3FBF8; }
.upload-box.has-image { padding: 1rem; }
.hidden-input { display: none; }
.preview-img {
  max-width: 100%;
  max-height: 260px;
  border-radius: 8px;
  margin-bottom: 0.75rem;
  object-fit: contain;
}
.upload-icon {
  width: 44px; height: 44px;
  border-radius: 10px;
  background: #E1F5EE;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 1rem;
}
.upload-icon i { font-size: 20px; color: #0F6E56; }
.upload-title { font-weight: 500; font-size: 16px; margin: 0 0 6px; }
.upload-sub { font-size: 13px; color: #6B6A64; margin: 0; }

.question-label { font-weight: 500; font-size: 15px; margin: 1.5rem 0 0.5rem; }
.question-input {
  display: flex; align-items: center; gap: 8px;
  border: 0.5px solid #D8D6CD;
  border-radius: 8px;
  padding: 0 12px; height: 36px;
  background: #FFFFFF;
}
.question-input i { font-size: 16px; color: #9B9A93; }
.question-text-input {
  flex: 1;
  border: none;
  outline: none;
  font-size: 13px;
  height: 100%;
  background: transparent;
}

.error-text { color: #C0392B; font-size: 13px; margin: 0.75rem 0 0; }

.submit-btn {
  margin-top: 1.25rem;
  background: #1D9E75;
  color: #FFFFFF;
  border: none;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  padding: 0 20px; height: 42px;
  border-radius: 8px;
  font-size: 14px; font-weight: 500;
  cursor: pointer;
  width: 100%;
}
.submit-btn:disabled { background: #A9D9C6; cursor: not-allowed; }
.submit-btn i { font-size: 16px; }

.model-badge {
  font-size: 12px;
  background: #E1F5EE; color: #0F6E56;
  padding: 4px 10px; border-radius: 999px;
  display: flex; align-items: center; gap: 4px;
  white-space: nowrap;
}
.model-badge i { font-size: 13px; }

.result-card {
  background: #FAFAF8;
  border: 0.5px solid #E4E2D9;
  border-radius: 12px;
  padding: 1.25rem;
}
.section-label { font-size: 13px; color: #6B6A64; margin: 0 0 8px; }
.placeholder-text { margin: 0; }
.diagnosis-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: #E1F5EE; color: #0F6E56;
  padding: 8px 14px; border-radius: 8px;
  font-size: 14px; font-weight: 500;
  margin-bottom: 1.25rem;
}
.diagnosis-pill i { font-size: 15px; }
.result-footer {
  display: flex; align-items: center; justify-content: flex-end;
  border-top: 0.5px solid #E4E2D9;
  padding-top: 12px;
}
.copy-btn {
  display: flex; align-items: center; gap: 6px;
  font-size: 13px; padding: 0 12px; height: 32px;
  background: #FFFFFF; border: 0.5px solid #D8D6CD; border-radius: 8px;
  cursor: pointer;
}
.copy-btn i { font-size: 14px; }

.history-section { margin-top: 1.5rem; padding-top: 1.25rem; border-top: 0.5px solid #E4E2D9; }
.history-title { margin-bottom: 0.75rem; }
.history-list {
  max-height: 320px;
  overflow-y: auto;
  padding-right: 4px;
}
.history-item {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 8px;
  background: #FAFAF8;
  border: 0.5px solid #E4E2D9;
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 8px;
}
.history-item:last-child { margin-bottom: 0; }
.history-q { font-size: 13px; margin: 0 0 4px; font-weight: 500; }
.history-a { font-size: 13px; margin: 0 0 4px; color: #0F6E56; }
.history-time { font-size: 11px; margin: 0; color: #9B9A93; }
.delete-btn {
  border: none; background: transparent; cursor: pointer;
  color: #C0392B; font-size: 15px; padding: 4px;
  flex-shrink: 0;
}

@media (max-width: 860px) {
  .page { padding: 1.5rem 1rem 3rem; }
  .grid { grid-template-columns: 1fr; gap: 1.5rem; }
  .app-header { margin-bottom: 1.5rem; }
}
</style>
