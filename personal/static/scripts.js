document.getElementById("predict-form").addEventListener("submit", async function (e) {
    e.preventDefault();

    const x = document.getElementById("x").value;
    const y = document.getElementById("y").value;
    const test_size = document.getElementById("test_size").value;
    const algorithm = document.getElementById("algorithm").value;

    const payload = {
        X: JSON.parse(x),
        y: JSON.parse(y),
        test_size: test_size,
        algorithm: algorithm
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            document.getElementById("result").innerHTML = `
                <strong>Predictions:</strong> ${JSON.stringify(data.predictions)}<br>
                <strong>Score:</strong> ${data.score}
            `;
        }
    } catch (error) {
        document.getElementById("result").innerText = "Something went wrong!";
        console.error(error);
    }
});
