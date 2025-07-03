document.getElementById("vqa-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const imageInput = document.getElementById("image-input");
  const questionInput = document.getElementById("question-input");
  const answerElem = document.getElementById("answer");

  if (imageInput.files.length === 0) {
    alert("Please select an image");
    return;
  }

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);
  formData.append("question", questionInput.value);

  try {
    // IMPORTANT: change this URL to your backend once deployed
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Network response was not ok");

    const data = await response.json();
    answerElem.textContent = data.answer;
  } catch (error) {
    answerElem.textContent = "Error: " + error.message;
  }
});
