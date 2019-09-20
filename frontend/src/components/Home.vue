<template>
  <div>
    <p>Home page</p>
    <p>Random number from backend: {{ randomNumber }}</p>
    <button @click="getRandom">New random number</button>
    <div>
      <p class="sample"><img :src="imgSrc1" width="480" height="270"></p>
      <img :src="imgSrc2">
      <img :src="imgSrc3">
      <p>{{ feature.slice(0,25) }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
export default {
  data () {
    return {
      randomNumber: 0,
      imgSrc1: require("../../../templates/images/image.jpg"),
      imgSrc2: require("../../../templates/images/face_detect.jpg"),
      imgSrc3: require("../../../templates/images/prepro.jpg")
    }
  },
  methods: {
    getRandom () {
      const path = 'http://localhost:5000/api/image'
      axios.get(path)
        .then(response => {
          this.randomNumber = response.data.randomNumber
          this.feature = response.data.feature
        })
        .catch(error => {
          console.log(error)
        })
    }
  },
  created () {
    this.getRandom()
  }
}
</script>

<style scoped>
  .sample img{
    width: 400px;
    height: auto;
  }
</style>>