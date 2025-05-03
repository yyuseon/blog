// 꽃잎 떨어지는 효과
document.addEventListener("DOMContentLoaded", function () {
    let petals = 30;
    const petalShapes = ["🌸", "💮", "❀", "✿"];
    for (let i = 0; i < petals; i++) {
        let petal = document.createElement("div");
        petal.className = "petal";
        petal.textContent = petalShapes[Math.floor(Math.random() * petalShapes.length)];
        document.body.appendChild(petal);
        petal.style.left = Math.random() * 100 + "vw";
        petal.style.animationDuration = (Math.random() * 4 + 4) + "s"; // 느리게: 최소 5초~최대 10초
        petal.style.fontSize = Math.random() * 10 + 10 + "px";
    }
});
