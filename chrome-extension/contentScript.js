console.log("Content Script with stuff", window.WebSocket);
// Inspired by https://github.com/RobbyChapman/chrome-socket-analyzer

var script = document.createElement("script");
script.src = chrome.runtime.getURL("contentScript2.js");
console.log(chrome.runtime.getURL("contentScript2.js"));
script.onload = function () {
  console.log("loaded");
  this.parentNode.removeChild(this);
};
console.log(document.head);
console.log(document.documentElement);
(document.head || document.documentElement).appendChild(script);

window.addEventListener(
  "RebroadcastExtensionMessage",
  function (evt) {
    chrome.runtime.sendMessage(evt.detail);
  },
  false
);
