


f = 0;
b = document.getElementById("a");
c = b.getContext("2d");
b.style.background = "white";
jsondata = []
l = [];
pulse = "";
//var lWidth = document.getElementsByName('pix')[0].getAttribute('value');
function iter() { if (l.length < 10) { l.push(one); } else { l.shift(); l.push(one); } }
function clrscr() { var r = confirm("¿Está seguro de eliminar todo el dibujo?"); if (r == true) { b.width = b.width; }; linewidth(); }
function color(colour) { c.beginPath(); c.strokeStyle = colour; c.fillStyle = colour; c.closePath(); }
function events() { b.onmousedown = md; b.onmouseup = mu; b.onmousemove = mv; }
function md(e) { one = c.getImageData(0, 0, b.width, b.height); iter(); img = c.getImageData(0, 0, b.width, b.height); sX = e.x; sY = e.y; pulse = "on"; }
function mu(e1) {
    eX = e1.x; eY = e1.y; pulse = "off";
    // if (item == 'c') { jsondata.push({ "Type": "line", "X0": sX, "Y0": sY, "X1": mX, "Y1": mY, "width": c.lineWidth, "color": c.strokeStyle }) }
    // if (item == 'a') { jsondata.push({ "Type": "rect", "X0": sX, "Y0": sY, "W": mX - sX, "H": mY - sY, "width": c.lineWidth, "color": c.strokeStyle, "fill": f }) }
    // if (item == 'b') { jsondata.push({ "Type": "circle", "X0": sX, "Y0": sY, "X1": mX, "Y1": mY, "width": c.lineWidth, "color": c.strokeStyle, "fill": f }) }
}
function mv(e2) { mX = e2.x ; mY = e2.y ; if (pulse == "on" && (item == 'e' || item == 'f')) { draw(); } else if (pulse == 'on') { c.putImageData(img, 0, 0); draw(); } }
function pencil() { events(); item = 'd'; }
function erase() { item = 'e'; events(); }
function spray() { item = 'f'; events(); }
function undo() { if (l.length >= 1) { b.width = b.width; c.putImageData(l[l.length - 1], 0, 0); l.pop(); } }
function draw() {
    if (item == 'a') { c.strokeRect(sX, sY, mX - sX, mY - sY); c.stroke(); if (f == 1) { c.fillRect(sX, sY, mX - sX, mY - sY); } }
    if (item == "b") { c.beginPath(); c.arc(Math.abs(mX + sX) / 2, Math.abs(mY + sY) / 2, Math.sqrt(Math.pow(mX - sX, 2) + Math.pow(mY - sY, 2)) / 2, 0, Math.PI * 2); c.stroke(); if (f == 1) { c.fill(); } c.closePath(); }
    if (item == "c") { c.beginPath(); c.moveTo(sX, sY); c.lineTo(mX, mY); c.stroke(); c.closePath(); }
    if (item == 'd') { c.moveTo(sX - 15, sY -120); c.lineTo(mX -15, mY- 120); c.stroke(); jsondata.push({ "Type": "pencil", "X0": sX, "Y0": sY, "X1": mX, "Y1": mY, "width": c.lineWidth, "color": c.strokeStyle }); sX = mX; sY = mY; }
    if (item == 'e') { c.clearRect(mX - 10, mY - 180, 30, 30); jsondata.push({ "Type": "eraser", "X0": mX - 25, "Y0": mY - 25 }) }
    if (item == 'f') {
        for (var i = 0; i < 20; i = i + 6) {
            jsondata.push({ "Type": "spray", "X0": mX, "Y0": mY, "color": c.fillStyle })
            c.fillRect(mX + i, mY + i, 1, 1);
            c.fillRect(mX - i, mY - i, 1, 1);
            c.fillRect(mX + i, mY - i, 1, 1);
            c.fillRect(mX - i, mY + i, 1, 1);
            c.fillRect(mX - i, mY, 1, 1);
            c.fillRect(mX, mY - i, 1, 1);
            c.fillRect(mX, mY + i, 1, 1);
            c.fillRect(mX + i, mY, 1, 1);
        }
    }
}
function fill() { f = 1; }
function strok() { f = 0; }
function linewidth() {
    c.beginPath();
    var s = document.getElementsByName('pix')[0].getAttribute('value');
    c.lineWidth = s; c.closePath(); lWidth = s;
}
function save() {



    $.post("/save/", {
        'data': document.getElementById("a").toDataURL("image/png").replace("image/png", "image/octet-stream"),
        'race': document.getElementById("race").value
    }, function (data, status) { alert("saved") });


}

var slider = document.getElementById("myRange");
var output = document.getElementById("demo");
output.innerHTML = slider.value;

slider.onchange = function () {
    output.innerHTML = this.value;
}
linewidth();