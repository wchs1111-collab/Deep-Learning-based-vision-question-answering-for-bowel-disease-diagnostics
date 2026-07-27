<script setup>
import { ref } from "vue";
import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";
const result = ref("");
const inputName = ref("");

async function getHello() {
  const res = await axios.get(`${BASE_URL}/api/hello`);
  result.value = JSON.stringify(res.data);
}

async function postEcho() {
  const res = await axios.post(`${BASE_URL}/api/echo`, {
    name: inputName.value
  });
  result.value = JSON.stringify(res.data);
}
</script>

<template>
  <div style="padding: 20px;">
    <h2>Vue 连接 FastAPI 示例</h2>

    <button @click="getHello">GET 请求</button>

    <div style="margin-top: 10px;">
      <input v-model="inputName" placeholder="输入内容" />
      <button @click="postEcho">POST 请求</button>
    </div>

    <p>返回结果：{{ result }}</p>
  </div>
</template>