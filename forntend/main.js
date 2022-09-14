
function readFileB () {
    // (A) GET SELECTED FILE
    let selected = document.getElementById("demoPickB").files[0] ;

    // (B) READ SELECTED FILE
    let reader = new FileReader();
    reader.addEventListener("load", () => {
        var data = new FormData();
        data.append("image", reader.result);

        var xhr = new XMLHttpRequest();
        xhr.withCredentials = false;

        xhr.open("POST", "http://127.0.0.1:5000/detect");

        xhr.onload = function() {
            let resultTag = document.getElementById("image");
            console.log(this.responseText)
            result = JSON.parse(this.response)
            resultTag.src = result["image"];
        };

        xhr.send(data);

    });
    
    reader.readAsDataURL(selected);


}
